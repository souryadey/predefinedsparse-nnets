#==============================================================================
# Utilities
#==============================================================================


def merge_dicts(a,b):
    '''
    If a and b have any common keys, their values must be lists. This function extends the lists in a by those in b
    If a has keys which b doesn't, their values need not be lists and are left intact
    If b has keys which a doesn't, their values need not be lists and are added to a
    Modifies a in-place, no need to return
    '''
    for k in b:
        if k in a:
            a[k].extend(b[k])
        else:
            a[k] = b[k]
