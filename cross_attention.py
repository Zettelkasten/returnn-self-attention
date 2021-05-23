from self_attention import make_lsh_hash_gen, apply_lsh_hash_gen


def _query_key_time_default(query_time_axis, key_time_axis):
  """
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :rtype: tuple[str,str]
  """
  assert (query_time_axis is None) == (key_time_axis is None)
  if query_time_axis is None:
    query_time_axis = 'stag:extern_data:classes'
    key_time_axis = 'stag:extern_data:data'
  assert query_time_axis.startswith('stag:')
  assert key_time_axis.startswith('stag:')
  return query_time_axis, key_time_axis


def _make_cross_attention_qkv(
  d, db, input, keys_input, output, num_heads=8, key_dim=64, value_dim=64,
  ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  d[output + '_query0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,query-T?,F|n*d_k]
  db[output + '_key0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_k]
  db[output + '_value0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [keys_input],
    'n_out': num_heads * value_dim, 'forward_weights_init': ff_init}  # [B,key-T,F|n*d_v]
  d[output + '_query_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_query0']}  # [B,query-T?,n,F|d_k]
  db[output + '_key_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_key0']}  # [B,key-T,n,F|d_k]
  db[output + '_value_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, value_dim),
    'from': [output + '_value0']}  # [B,key-T,n,F|d_v]
  d[output + '_query'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_query_unnamed']}  # [B,query-T?,n,F|d_k]
  db[output + '_key'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_key_unnamed']}  # [B,key-T,n,F|d_k]
  db[output + '_value'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_value_unnamed']}  # [B,key-T,n,F|d_v]


def add_vanilla_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  """
  Add a cross-attention layer.

  :param dict[str, Any] d:
  :param dict[str, Any] db:
  :param str input:
  :param str keys_input:
  :param str output:
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :param int num_heads:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param str ff_init:
  """
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)

  assert keys_input.startswith('base:')
  keys_input = keys_input[len('base:'):]

  # Create query, key and value
  _make_cross_attention_qkv(
    d=d, db=db, input=input, keys_input=keys_input, output=output, num_heads=num_heads, key_dim=key_dim,
    value_dim=value_dim, ff_init=ff_init)

  # Calculate the energies + weights
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', 'base:' + output + '_key'], 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': query_time_axis + '?', 'var2': key_time_axis}  # [B,n,query-T?,key-T]
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': key_time_axis,
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': True}  # [B,n,query-T?,key-T]
  d[output + '_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights'],
    'dropout': dropout}  # [B,n,query-T?,key-T]

  d[output + '_output_named'] = {
    'class': 'dot', 'from': [output + '_weights_drop', 'base:' + output + '_value'], 'red1': key_time_axis,
    'red2': key_time_axis, 'var1': query_time_axis + '?', 'var2': 'static:-1'}  # [B,n,query-T?,d_v]
  d[output + '_output'] = {
    'class': 'name_axis', 'from': [output + '_output_named'], 'axis': 'stag:att-heads',
    'description': None}  # [B,n,query-T?,d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output']}  # [B,query-T?,F|n*d_v]

  # there is a bug in HDF dump layer when naming static dimensions. Expose this here to extract the att weights.
  d[output + '_weights_unnamed'] = {
    'class': 'name_axis', 'from': [output + '_weights'], 'axis': 'stag:att-heads',
    'description': None}  # [B,n,query-T?,key-T]


def add_full_lsh_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  num_hashes=14, num_rounds=1, mask_current_value=float(-10**5), mask_different_hashes=True, debug_print=False):
  """
  Add a cross-attention layer with masking as in the LSH case.
  This way, you can e.g. train a system using LSH attention, but then do search using this.

  :param dict[str, Any] d:
  :param dict[str, Any] db:
  :param str input:
  :param str keys_input:
  :param str output:
  :param None|str query_time_axis:
  :param None|str key_time_axis:
  :param int num_heads:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param str ff_init:
  :param int num_hashes:
  :param int num_rounds:
  :param float mask_current_value:
  :param bool mask_different_hashes:
  :param bool debug_print:
  """
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)

  # build standard self-attention
  assert keys_input.startswith('base:')
  add_vanilla_cross_attention_layer(
    d=d, db=db, input=input, keys_input=keys_input, output=output, query_time_axis=query_time_axis,
    key_time_axis=key_time_axis, num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, dropout=dropout,
    ff_init=ff_init)  # [B,n,r,d_k,F|d_h]

  # hash keys and queries
  assert mask_different_hashes, 'can just call add_vanilla_cross_attention_layer(..) instead'
  make_lsh_hash_gen(
    db, output + '_hash_gen', key_dim=key_dim, num_hashes=num_hashes, num_heads=num_heads, num_rounds=num_rounds,
    ff_init=ff_init)  # [B,n,r,d_k,F|d_h]
  apply_lsh_hash_gen(
    d, input=output + '_query', hash_gen_input='base:' + output + '_hash_gen', output=output + '_query_hash',
    time_axis=query_time_axis, hash_mask_value=None)  # [B,n,r,query-T?] :: d_h
  apply_lsh_hash_gen(
    db, input=output + '_key', hash_gen_input=output + '_hash_gen', output=output + '_key_hash',
    time_axis=key_time_axis, hash_mask_value=None)  # [B,n,r,key-T] :: d_h
  assert num_rounds == 1, 'not implemented yet otherwise'

  # build and apply additional energy mask
  d[output + '_energy_mask_rounds'] = {
    'class': 'compare', 'from': [output + '_query_hash', 'base:' + output + '_key_hash'],
    'kind': 'equal'}  # [B,n,r,T-query?,T-key]
  d[output + '_energy_mask'] = {
    'class': 'squeeze', 'axis': 'stag:att-round', 'from': [output + '_energy_mask_rounds']}  # [B,n,T-query?,T-key]
  assert (output + '_energy') in d
  d[output + '_energy_unmasked'] = d[output + '_energy']  # [B,n,query-T?,key-T]
  d[output + '_energy'] = {
    'class': 'switch', 'condition': output + '_energy_mask',
    'true_from': output + '_energy_unmasked', 'false_from': mask_current_value}  # [B,n,query-T?,key-T]

  if debug_print:
    for name in [output + n for n in [
      '_query', '_query_hash', '_energy_mask', '_energy_unmasked', '_energy', '_weights']] + [
      'base:' + output + '_key', 'base:' + output + '_key_hash']:
      if name.startswith('base:'):
        name = name[len('base:'):]
        assert name in db and name + '_orig' not in db
        db[name + '_orig'] = db[name]
        db[name] = {'class': 'print', 'from': [name + '_orig']}
      else:
        assert name in d and name + '_orig' not in d
        d[name + '_orig'] = d[name]
        d[name] = {'class': 'print', 'from': [name + '_orig']}


def add_lsh_cross_attention_layer(
  d, db, input, keys_input, output, query_time_axis=None, key_time_axis=None, *,
  num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0, num_hashes, key_chunk_size, query_chunk_size,
  key_chunks_before=None, key_chunks_after=None,
  ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  small_mask_value=float(-10**5), mask_different_hashes=True, fallback_mode, allow_duplicate_attention=False,
  debug_print=False):
  query_time_axis, key_time_axis = _query_key_time_default(query_time_axis, key_time_axis)
  assert keys_input.startswith('base:')
  keys_input = keys_input[len('base:'):]

  # Create query, key and value
  _make_cross_attention_qkv(
    d=d, db=db, input=input, keys_input=keys_input, output=output, num_heads=num_heads, key_dim=key_dim,
    value_dim=value_dim, ff_init=ff_init)

  # Accumulate the queries
  queries_input = output + '_query_accum'
  d[output + '_query_accum'] = {'class': 'cum_concat', 'from': [output + '_query'], 'axis': query_time_axis}

  # Apply lsh hashing for all queries
  from lsh import add_lsh_attention_layer
  add_lsh_attention_layer(
    d=d, queries_input=queries_input, keys_input='base:' + output + '_key',
    values_input='base:' + output + '_value', output=output, query_time_axis='stag:rec-history',
    key_time_axis=key_time_axis, num_heads=num_heads, num_rounds=num_rounds, key_dim=key_dim, value_dim=value_dim,
    dropout=dropout, num_hashes=num_hashes, query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size,
    key_chunks_before=key_chunks_before, key_chunks_after=key_chunks_after, ff_init=ff_init,
    small_mask_value=small_mask_value, mask_different_hashes=mask_different_hashes, fallback_mode=fallback_mode,
    allow_duplicate_attention=allow_duplicate_attention, debug_print=debug_print)

  # Select the context vector for the query we actually want
  d[output + '_att'] = {'class': 'gather', 'from': [output + '_att_all'], 'position': ':i', 'axis': 'stag:rec-history'}
