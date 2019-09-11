def LORD(stream,alpha):
    # Inputs: stream - array of p-values, alpha - target FDR level
    # Output: array of indices k such that the k-th p-value corresponds to a discovery
    
    gamma = lambda t: 6/(math.pi*t)**2
    w_0 = alpha/2
    rejections = []
    alpha_t = gamma(1)*w_0
    for t in range(1,n+1):
        # Offset by one since indexing by 1 for t
        p_t = stream[t-1] 
        
        if p_t < alpha_t:
            rejections.append(t)

        next_alpha_t = gamma(t+1)*w_0 + alpha*sum([gamma(t+1-rej) for rej in rejections])
        # Check if tau_1 exists
        if len(rejections)>0:
            next_alpha_t -= gamma(t+1-rejections[0])*w_0
        

        # Update alpha
        alpha_t = next_alpha_t
    # Shift rejections since the rejections are 1-indexed
    shifted_rej = [rej-1 for rej in rejections]
    return shifted_rej