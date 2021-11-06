import tensorflow as tf

def define_loss(network, params):
    """
    https://arxiv.org/pdf/1904.02107.pdf

    Create the loss functions.

    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """

    x = network['x']
    # reconstructed x 
    x_decode = network['x_decode']
    if params['model_order'] == 1:
        dz = network['dz']
        dz_predict = network['dz_predict']
        dx = network['dx']
        dx_decode = network['dx_decode']
    else:
        ddz = network['ddz']
        ddz_predict = network['ddz_predict']
        ddx = network['ddx']
        ddx_decode = network['ddx_decode']
    sindy_coefficients = params['coefficient_mask']*network['sindy_coefficients']

    losses = {}
    # get norm two of x reconstruction
    losses['decoder'] = tf.reduce_mean((x - x_decode)**2)
    if params['model_order'] == 1:
        # get lost in z reduced space = SINDy loss in z'
        losses['sindy_z'] = tf.reduce_mean((dz - dz_predict)**2)
        # get lost in x space = SINDy loss in x'
        losses['sindy_x'] = tf.reduce_mean((dx - dx_decode)**2)
    else:
        losses['sindy_z'] = tf.reduce_mean((ddz - ddz_predict)**2)
        losses['sindy_x'] = tf.reduce_mean((ddx - ddx_decode)**2)
    
    # SINDy regularization 
    losses['sindy_regularization'] = tf.reduce_mean(tf.abs(sindy_coefficients))
    
    # weighted total loss including SINDy regularization 
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_sindy_z'] * losses['sindy_z'] \
           + params['loss_weight_sindy_x'] * losses['sindy_x'] \
           + params['loss_weight_sindy_regularization'] * losses['sindy_regularization']

    # weighted total loss without SINDy regularization 
    # still to figure out why the hell we care about loss_refinemen
    loss_refinement =   params['loss_weight_decoder'] * losses['decoder'] \
                        + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                        + params['loss_weight_sindy_x'] * losses['sindy_x']

    return loss, losses, loss_refinement

def sindy_library_tf(z, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library.

    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.

    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    library = [tf.ones(tf.shape(z)[0])]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))

    return tf.stack(library, axis=1)