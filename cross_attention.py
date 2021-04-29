
def add_vanilla_cross_attention_layer(
  d, input, output, keys_input, query_time_axis=None, key_time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  """
  Add a cross-attention layer.

  :param dict[str, Any] d:
  :param str input:
  :param str output:
  :param str keys_input:
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :param int num_heads:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param str ff_init:
  """
  assert (query_time_axis is None) == (key_time_axis is None)
  if query_time_axis is None:
    query_time_axis = 'stag:extern_data:classes'
    key_time_axis = 'stag:extern_data:data'
  assert query_time_axis.startswith('stag:')
  assert key_time_axis.startswith('stag:')

  # Create query, key and value
  d[output + '_query0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,query-T?,F|n*d_k]
  d[output + '_key0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_k]
  d[output + '_value0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * value_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_v]
  d[output + '_query'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_query0']}  # [B,query-T?,n,F|d_k]
  d[output + '_key'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_key0']}  # [B,key-T,n,F|d_k]
  d[output + '_value'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, value_dim),
    'from': [output + '_value0']}  # [B,key-T,n,F|d_v]

  # Calculate the energies
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', output + '_key'], 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': query_time_axis + '?', 'var2': key_time_axis}  # [B,n,query-T?,key-T]

  # If past_only=True, do not apply a time mask here, as we apply our own masking using energy_mask.
  # If we would apply additional masking here, we would mask away all keys for queries that are unmasked, giving
  # attention weights NaN for these queries. Even though these are masked away later in the forward pass, the gradient
  # can still become NaN.
  # If past_only=False, do apply the normal time mask.
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': key_time_axis,
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': True}  # [B,n,query-T?,key-T]
  d[output + '_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights'],
    'dropout': dropout}  # [B,n,query-T?,key-T]

  d[output + '_output'] = {
    'class': 'dot', 'from': [output + '_weights_drop', output + '_value'], 'red1': key_time_axis, 'red2': key_time_axis,
    'var1': query_time_axis + '?', 'var2': 'static:-1'}  # [B,n,query-T?,key-T]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output']}  # [B,query-T?,F|n*d_v]
