import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class Flatten(nn.Module):
    r"""Reshapes the input tensor as a 2d tensor, where the size of the first (batch) dimension is preserved.

    Inputs:
        - **inputs** (batch, num_dim1, num_dim1,...): tensor containing input features

    Outputs:
        - **outputs** (batch, num_dim1*num_dim2*...): tensor containing the output
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


def compute_codes_loss(codes, nmod, lin, loss_fn, codes_targets, mu, lambda_c):
    r"""Function that computes the code loss

    Inputs:
        - **codes** (batch, num_features): outputs of the linear modules
        - **nmod** (nn.Module): non-linear module downstream from linear module
        - **lin** (nn.Conv2d or nn.Lineae):  linear module
        - **loss_fn**: loss function computed on outputs
        - **codes_targets** (batch, num_features): target to which codes have to be close (in L2 distance)
        - **lambda_c** (scalar): Lagrance muliplier for code loss function
    Outputs:
        - **loss**: loss
    """
    output = lin(nmod(codes))
    loss = (1/mu)*loss_fn(output) + F.mse_loss(codes, codes_targets)
    if lambda_c>0.0:
        loss += (lambda_c/mu)*codes.abs().mean()
    return loss


def update_memory(As, Bs, inputs, codes, model_mods, eta=0.0):
    r"""Updates the bookkeeping matrices using codes as in Mairal et al. (2009)

    Inputs:
        - **As** (list): list of codes autocovariance matrices
        - **Bs** (list): list of cross-covariance matrices between codes and model outputs
        - **inputs** (batch, num_features): tensor of inputs
        - **codes** (batch, num_features): tensor of codes (i.e. intermediate layer activations)
        - **model_mods** (list): list of model modules
        - **eta** (scalar): linear filtering factor

    Outputs:
        - **As** (list): list of updated codes autocovariance matrices
        - **Bs** (list): list of updated cross-covariance matrices between codes and model outputs
    """
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    with torch.no_grad():
        id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
        for i, (idx, c_in, c_out) in enumerate(zip(id_codes, [x]+codes[:-1], codes)):
            try:
                nmod = model_mods[idx-1]
            except IndexError:
                nmod = lambda x: x

            a = nmod(c_in)
            if eta == 0.0:
                As[i] += a.t().mm(a)
                Bs[i] += c_out.t().mm(a)
            else:
                As[i] = (1-eta)*As[i] + eta*a.t().mm(a)
                Bs[i] = (1-eta)*Bs[i] + eta*c_out.t().mm(a)
    return As, Bs


def update_hidden_weights_bcd_(model_mods, As, Bs, lambda_w, max_iter=1):
    r"""Use BCD to update weights of intermediate modules
    """
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for i, A, B in zip(id_codes, As, Bs):
        model_mods[i].weight.data = BCD(model_mods[i].weight.data, A, B, lambda_w, max_iter=max_iter)


def BCD(w, A, B, lambda_w, eps=1e-3, max_iter=20, return_errors=False):
    r"""BCD algorithm to update weights w based on bookkeeping matrices A and B
        lambda_w is referenced to A_jj in every column
    """
    B = B.div(A.diag() + 1e-10)
    A = A.div(A.diag() + 1e-10)

    errors = []
    with torch.no_grad():
        for i in range(max_iter):
            w_pre = w.clone()
            error = 0
            for j in range(A.shape[1]):
                delta_j = (B[:,j] - w.mv(A[:,j]))
                w[:,j].add_(delta_j)
                #  u_j /= max(u_j.norm(), 1.0) # This was in Mairal2009, but assumes that B has spectral radius smaller than A
                # Shrinkage step (sparsity regularizer)
                if lambda_w > 0.0:
                    sign_w = w[:,j].sign()
                    w[:,j].abs_().add_(-lambda_w).clamp_(min=0.0).mul_(sign_w)
                error += delta_j.abs().mean().item()
            errors.append(error)
            # Converged is there is no change between steps
            if (w - w_pre).abs().max().item() < eps:
                break

    if return_errors:
        return w, errors
    else:
        return w


def post_processing_step(model, data, target, criterion, n_iter=1):
    with torch.no_grad():
        output, codes = get_codes(model, data)

    update_last_layer_(model[-1], codes[-1], target, criterion, n_iter=n_iter)


def insert_mod(model_mods, mod, has_codes):
    "If a mod is not empty, close it, include it, and start a new mod"
    if len(mod) == 1:
        model_mods.add_module(str(len(model_mods)), mod[0])
        model_mods[-1].has_codes = has_codes
    elif len(mod) > 1:
        model_mods.add_module(str(len(model_mods)), mod)
        model_mods[-1].has_codes = has_codes
    mod = nn.Sequential()
    return mod


def mark_code_mods_(model, module_types=None):
    '''Marks the modules of model of type module_types for code generation, i.e. it sets has_codes to True
        If a module already has has_codes set to True, it will be left True'''
    if module_types is None:
        module_types = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d]

    for m in list(model.features) + [Flatten()] + list(model.classifier):
        if any([isinstance(m, t) for t in module_types]):
            m.has_codes = True
        if not hasattr(m, 'has_codes'):
            m.has_codes = False


def get_mods(model, optimizer=None, optimizer_params={}, scheduler=None, data_parallel=False, module_types=None):
    '''Returns all the modules in a nn.Sequential alternating linear and non-linear modules
        Arguments:
            optimizer: if not None, each module will be given an optimizer of the indicated type
            with parameters in the dictionary optimizer_params
    '''
    mark_code_mods_(model, module_types=module_types)

    model_mods = nn.Sequential()
    if hasattr(model, 'n_inputs'):
        model_mods.n_inputs = model.n_inputs

    nmod, lmod = nn.Sequential(), nn.Sequential()
    for m in list(model.features) + [Flatten()] + list(model.classifier):
        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            nmod = insert_mod(model_mods, nmod, has_codes=False)
            lmod.add_module(str(len(lmod)), m)
        else:
            lmod = insert_mod(model_mods, lmod, has_codes=True)
            nmod.add_module(str(len(nmod)), m)

    insert_mod(model_mods, nmod, has_codes=False)
    insert_mod(model_mods, lmod, has_codes=True)

    # Last layer that generates codes is lumped together with adjacent modules to produce the last layer
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

    model_tmp = model_mods[:id_codes[-2]+1]
    model_tmp.add_module(str(len(model_tmp)), model_mods[id_codes[-2]+1:])
    model_tmp[-1].has_codes = False
    model_mods = model_tmp

    if optimizer is not None:
        # Include an optimizer in modules with codes
        for m in model_mods:
            if m.has_codes:
                m.optimizer = getattr(optim, optimizer)(m.parameters(), **optimizer_params)
                if scheduler is not None:
                    m.scheduler = LambdaLR(m.optimizer, lr_lambda=scheduler)

        # Add optimizer to the last layer
        model_mods[-1].optimizer = getattr(optim, optimizer)(model_mods[-1].parameters(), **optimizer_params)
        if scheduler is not None:
            m.scheduler = LambdaLR(m.optimizer, lr_lambda=scheduler)

    if data_parallel:
        data_parallel_mods_(model_mods)

    return model_mods


def data_parallel_mods_(model_mods):
    for i,m in enumerate(model_mods):
        model_mods[i] = torch.nn.DataParallel(m)
        model_mods[i].has_codes = m.has_codes
        if hasattr(m, 'optimizer'):
            model_mods[i].optimizer = m.optimizer
        if hasattr(m, 'scheduler'):
            model_mods[i].scheduler = m.scheduler


def get_codes(model_mods, inputs):
    '''Runs the architecture forward using `inputs` as inputs, and returns outputs and intermediate codes
    '''
    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    # As codes only return outputs of linear layers
    codes = []
    for m in model_mods:
        x = m(x)
        if hasattr(m, 'has_codes') and getattr(m, 'has_codes'):
            codes.append(x.clone())
    # Do not include output of very last linear layer (not counted among codes)
    return x, codes


def update_codes(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        codes[-l].requires_grad_(True)
        optimizer = optim.SGD([codes[-l]], lr=lr, momentum=0.9, nesterov=True)
        codes_initial = codes[-l].clone()

        if idx+1 in id_codes:
            nmod = lambda x: x
            lin = model_mods[idx+1]
        else:
            try:
                nmod = model_mods[idx+1]
            except IndexError:
                nmod = lambda x: x
            try:
                lin = model_mods[idx+2]
            except IndexError:
                lin = lambda x: x

        if l == 1:  # last layer
            loss_fn = lambda x: criterion(x, targets)
        else:       # intermediate layers
            loss_fn = lambda x: mu*F.mse_loss(x, codes[-l+1].detach())

        for it in range(n_iter):
            optimizer.zero_grad()
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            loss.backward()
            optimizer.step()

    return codes


def update_last_layer_(mod_out, inputs, targets, criterion, n_iter):
    for it in range(n_iter):
        mod_out.optimizer.zero_grad()
        outputs = mod_out(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        mod_out.optimizer.step()


def update_hidden_weights_adam_(model_mods, inputs, codes, lambda_w, n_iter):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    for idx, c_in, c_out in zip(id_codes, [x]+codes[:-1], codes):
        lin = model_mods[idx]
        if idx >= 1 and not idx-1 in id_codes:
            nmod = model_mods[idx-1]
        else:
            nmod = lambda x: x

        for it in range(n_iter):
            lin.optimizer.zero_grad()
            loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            lin.optimizer.step()


def scheduler_step(model_mods):
    for m in model_mods:
        if hasattr(m, 'scheduler'):
            m.scheduler.step()


# ------------------------------------------------------------------------
# Non-diff
# ------------------------------------------------------------------------
def update_hidden_weights_nondiff_(model_mods, inputs, codes, lambda_w, n_iter):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]

    if hasattr(model_mods, 'n_inputs'):
        x = inputs.view(-1, model_mods.n_inputs)
    else:
        x = inputs

    for idx, c_in, c_out in zip(id_codes, [x]+codes[:-1], codes):
        lin = model_mods[idx]
        if idx >= 1 and not idx-1 in id_codes:
            nmod = model_mods[idx-1]
        else:
            nmod = lambda x: x

        for it in range(n_iter):
            lin.optimizer.zero_grad()
            # Hinge loss
            loss = F.relu(0.0-lin(nmod(c_in))*c_out.detach().sign()).mean()
            #  loss = F.mse_loss(lin(nmod(c_in)), c_out.detach())
            if lambda_w > 0.0:
                loss += lambda_w*lin.weight.abs().mean()
            loss.backward()
            lin.optimizer.step()



def get_mods_nondiff(model, optimizer=None, optimizer_params={}, scheduler=None, data_parallel=False):
    model_mods = get_mods(model, optimizer=optimizer, optimizer_params=optimizer_params, scheduler=scheduler, data_parallel=False)
    model_ret = model_mods[:-1]
    for mod in model_mods[-1]:
        model_ret.add_module(str(len(model_ret)), mod)

    # Add optimizer to last layer
    model_mods[-1].has_codes = False
    model_mods[-1].optimizer = getattr(optim, optimizer)(model_mods[-1].parameters(), **optimizer_params)
    if scheduler is not None:
        m.scheduler = LambdaLR(m.optimizer, lr_lambda=scheduler)

    if data_parallel:
        data_parallel_mods_(model_ret)

    return model_ret


def update_codes_nondiff(codes, model_mods, targets, criterion, mu, lambda_c, n_iter, lr):
    id_codes = [i for i,m in enumerate(model_mods) if hasattr(m, 'has_codes') and getattr(m, 'has_codes')]
    for l in range(1, len(codes)+1):
        idx = id_codes[-l]

        codes_initial = codes[-l].clone().detach()
        codes[-l].requires_grad_(True)
        optimizer = optim.SGD([codes[-l]], lr=lr, momentum=0.9, nesterov=True)

        if l == 1:  # last layer
            nmod = model_mods[idx+1]
            try:
                lin = model_mods[idx+2]
            except IndexError:
                lin = lambda x: x
            loss_fn = lambda x: criterion(x, targets)
        else:       # intermediate layers
            idx_next = id_codes[-l+1]

            nmod = model_mods[idx+1:idx_next]
            lin = model_mods[idx_next]

            loss_fn = lambda x: mu*F.mse_loss(x, codes[-l+1].detach())# + 1.0*torch.min(F.relu(1+x), F.relu(1-x)).mean()
            #+ 0.1*(torch.min(x.abs(), (1-x).abs())).mean()

        for it in range(n_iter):
            optimizer.zero_grad()
            loss = compute_codes_loss(codes[-l], nmod, lin, loss_fn, codes_initial, mu, lambda_c)
            loss.backward()
            optimizer.step()

    return codes
