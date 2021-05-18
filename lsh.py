from self_attention import make_lsh_hash_gen, apply_lsh_hash_gen, argsort_eval, normalize_eval


def add_lsh_attention_layer(
  d, queries_input, keys_input, values_input, output, *,
  query_time_axis, key_time_axis, num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0,
  num_hashes, query_chunk_size, key_chunk_size, key_chunks_before=None, key_chunks_after=None,
  ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  small_mask_value=float(-10**5),
  past_only=None, mask_current=None, mask_different_hashes=True, allow_duplicate_attention=False,
  debug_print=False):
  """
  Computes LSH attention for an entire sequence.

  :param dict[str,dict] d:
  :param str queries_input:
  :param str keys_input:
  :param str values_input:
  :param str output:
  :param str query_time_axis:
  :param str key_time_axis: key and value time axis
  :param int num_heads:
  :param int num_rounds:
  :param int key_dim:
  :param int value_dim:
  :param float dropout:
  :param int num_hashes:
  :param int query_chunk_size:
  :param int key_chunk_size:
  :param int|None key_chunks_before:
  :param int|None key_chunks_after:
  :param str ff_init:
  :param float small_mask_value:
  :param None|bool past_only: for self attention
  :param None|bool mask_current: for self attention
  :param bool mask_different_hashes:
  :param bool allow_duplicate_attention:
  :param bool debug_print:
  """
  assert query_time_axis.startswith('stag:') and key_time_axis.startswith('stag:')
  self_attention = query_time_axis == key_time_axis
  assert self_attention == (past_only is not None) == (mask_current is not None)
  if key_chunks_before is None:
    key_chunks_before = 1
  if key_chunks_after is None:
    key_chunks_after = 0 if past_only else 1
  assert key_chunks_before >= 0 and key_chunks_after >= 0
  hash_mask_value = 2 ** 31 - 1
  assert hash_mask_value > num_hashes

  def chunk_query_sequence(name, pad_value):
    """
    :param str name:
    :param float pad_value:
    """
    d[output + '_sorted_chunked_%s_unnamed' % name] = {
      'class': 'split_dims', 'from': [output + '_sorted_%s' % name], 'pad_value': pad_value,
      'axis': query_time_axis, 'dims': [-1, query_chunk_size]}  # [B,n,r,query-chunk,query-window,F] :: query-time
    d[output + '_sorted_chunked_%s' % name] = {
      'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['query-chunk', 'query-window'],
      'from': [
        output + '_sorted_chunked_%s_unnamed' % name]}  # [B,n,r,query-chunk,query-window,F] :: query-time

  def chunk_key_sequence(name, pad_value):
    """
    :param str name:
    :param float pad_value:
    """
    d[output + '_sorted_chunked_%s_unnamed' % name] = {
      'class': 'split_dims', 'from': [output + '_sorted_%s' % name], 'pad_value': pad_value,
      'axis': key_time_axis, 'dims': [-1, key_chunk_size]}  # [B,n,r,key-chunk,key-window,F] :: key-time
    d[output + '_sorted_chunked_%s' % name] = {
      'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
      'from': [output + '_sorted_chunked_%s_unnamed' % name]}  # [B,n,r,key-chunk,key-window,F] :: key-time

  def stack_chunked_key_sequence(name):
    """
    :param str name:
    """
    d[output + '_sorted_chunked_stacked_%s_unflattened' % name] = {
      'class': 'gather', 'from': [output + '_sorted_chunked_%s' % name], 'position': output + '_query_chunk_alignment',
      'axis': 'stag:key-chunk'}  # [B,n,r,query-chunk,key-chunk-offset,key-window,F]
    d[output + '_sorted_chunked_stacked_%s_unnamed' % name] = {
      'class': 'merge_dims', 'from': [output + '_sorted_chunked_stacked_%s_unflattened' % name], 'keep_order': True,
      'axes': ['stag:key-chunk-offset', 'stag:key-window']}  # [B,n,r,query-chunk,key-chunk-offset*key-window,F]
    d[output + '_sorted_chunked_stacked_%s' % name] = {
      'class': 'name_axis', 'from': [output + '_sorted_chunked_stacked_%s_unnamed' % name],
      'axis': 'stag:query-chunk+1',
      'description': 'stacked-key-window'}  # [B,n,r,query-chunk,stacked-key-window,F]

  assert allow_duplicate_attention, 'not implemented'

  # Hash the queries and keys
  make_lsh_hash_gen(
    d, output + '_hash_gen', key_dim=key_dim, num_hashes=num_hashes, num_heads=num_heads, num_rounds=num_rounds,
    ff_init=ff_init)  # [B,n,r,d_k,F|d_h]
  apply_lsh_hash_gen(
    d, input=queries_input, hash_gen_input=output + '_hash_gen', output=output + '_queries_hashed',
    time_axis=query_time_axis)  # [B,query-time,n,r] :: d_h
  apply_lsh_hash_gen(
    d, input=keys_input, hash_gen_input=output + '_hash_gen', output=output + '_keys_hashed',
    time_axis=key_time_axis)  # [B,key-time,n,r] :: d_h

  # Compute a permutation by looking at the unsorted hashes
  d[output + '_sorted_queries_orig_indices'] = {
    'class': 'eval', 'eval': argsort_eval % query_time_axis,
    'from': [output + '_queries_hashed']}  # [B,sorted-query-time,n,r] :: query-time
  d[output + '_sorted_keys_orig_indices'] = {
    'class': 'eval', 'eval': argsort_eval % key_time_axis,
    'from': [output + '_keys_hashed']}  # [B,sorted-key-time,n,r] :: key-time
  chunk_query_sequence('queries_orig_indices', pad_value=hash_mask_value)  # [B,n,r,query-chunk,query-window] :: query-time  # noqa
  chunk_key_sequence('keys_orig_indices', pad_value=hash_mask_value)  # [B,n,r,key-chunk,key-window] :: key-time
  stack_chunked_key_sequence('keys_orig_indices')  # [B,n,r,query-chunk,stacked-key-window] :: key-time

  # Invert permutation to undo sorting later
  d[output + '_queries_all_indices'] = {
    'class': 'range_in_axis', 'from': [queries_input], 'axis': query_time_axis,
    'keepdims': False}  # [query-time] :: query-time
  d[output + '_queries_sort_indices'] = {
    'class': 'scatter_nd', 'from': [output + '_queries_all_indices'],
    'position': output + '_sorted_queries_orig_indices', 'position_axis': query_time_axis,
    'output_dim_via_time_from': output + '_sorted_queries_orig_indices'}  # [B,n,r,query-time] :: sorted-query-time

  # Sort hashes themselves
  d[output + '_sorted_queries_hashed'] = {
    'class': 'gather', 'from': [output + '_queries_hashed'], 'axis': query_time_axis,
    'position': output + '_sorted_queries_orig_indices'}  # [B,sorted-query-time,n,r] :: d_h
  d[output + '_sorted_keys_hashed'] = {
    'class': 'gather', 'from': [output + '_keys_hashed'], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r] :: d_h
  chunk_query_sequence('queries_hashed', pad_value=hash_mask_value)  # [B,n,r,query-chunk,query-window] :: d_h
  chunk_key_sequence('keys_hashed', pad_value=hash_mask_value)  # [B,n,r,key-chunk,key-window] :: d_h
  stack_chunked_key_sequence('keys_hashed')  # [B,n,r,query-chunk,stacked-key-window] :: d_h

  # Sort the queries, keys, values by applying the permutation
  d[output + '_sorted_queries_unscaled'] = {
    'class': 'gather', 'from': [queries_input], 'axis': query_time_axis,
    'position': output + '_sorted_queries_orig_indices'}  # [B,sorted-query-time,n,r,F|d_k]
  d[output + '_sorted_queries'] = {
    'class': 'eval', 'eval': '%s * source(0)' % (key_dim ** -0.5),
    'from': [output + '_sorted_queries_unscaled']}  # [B,sorted-query-time,n,r,F|d_k]
  d[output + '_sorted_keys'] = {
    'class': 'gather', 'from': [keys_input], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r,F|d_k]
  d[output + '_sorted_values'] = {
    'class': 'gather', 'from': [values_input], 'axis': key_time_axis,
    'position': output + '_sorted_keys_orig_indices'}  # [B,sorted-key-time,n,r,F|d_v]

  # Chunk the sorted queries and keys and values
  chunk_query_sequence('queries', pad_value=0.0)  # [B,n,r,query-chunk,query-window,F|d_k]
  chunk_key_sequence('keys', pad_value=0.0)  # [B,n,r,key-chunk,key-window,F|d_k]
  chunk_key_sequence('values', pad_value=0.0)  # [B,n,r,key-chunk,key-window,F|d_v]

  # Compute chunk alignment from query chunks to a fixed-sized set of key chunks
  d[output + '_query_chunk_alignment_center'] = {
    'class': 'range_in_axis', 'axis': 'stag:query-chunk', 'keepdims': False,
    'from': [output + '_sorted_chunked_queries']}  # [B,n,r,query-chunk] :: key-chunk
  d[output + '_query_chunk_alignment_offset_unnamed'] = {
    'class': 'range', 'start': -key_chunks_before, 'delta': 1, 'limit': key_chunks_after + 1}  # [key-chunk-offset]
  d[output + '_query_chunk_alignment_offset'] = {
    'class': 'name_axis', 'from': [output + '_query_chunk_alignment_offset_unnamed'], 'axis': 'F',
    'description': 'key-chunk-offset'}  # [key-chunk-offset]
  d[output + '_query_chunk_alignment_unbounded'] = {
    'class': 'combine', 'from': [output + '_query_chunk_alignment_center', output + '_query_chunk_alignment_offset'],
    'kind': 'add'}  # [B,n,r,query-chunk,key-chunk-offset] :: key-chunk
  d[output + '_key_chunk_count'] = {
    'class': 'length', 'from': [output + '_sorted_chunked_keys']}  # [B]
  d[output + '_query_chunk_alignment'] = {
    'class': 'eval', 'from': [output + '_query_chunk_alignment_unbounded', output + '_key_chunk_count'],
    'eval': 'tf.math.floormod(source(0), source(1))'}  # [B,n,r,query-chunk,key-chunk-offset] :: key-chunk

  # Collect stacked key and value chunks
  stack_chunked_key_sequence('keys')  # [B,n,r,query-chunk,stacked-key-window,F|d_k]
  stack_chunked_key_sequence('values')  # [B,n,r,query-chunk,stacked-key-window,F|d_v]

  # Compute chunked masking
  masking_layers_from = []
  if past_only:
    d[output + '_sorted_chunked_mask_past_only'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_queries_orig_indices', output + '_sorted_chunked_stacked_keys_orig_indices'],
      'kind': 'greater_equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    masking_layers_from.append(output + '_sorted_chunked_mask_past_only')
  if mask_different_hashes:
    # Masking valid positions is not necessary in this case as invalid positions will be masked with a special
    # hash value
    d[output + '_sorted_chunked_mask_matching_hash'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_queries_hashed', output + '_sorted_chunked_stacked_keys_hashed'],
      'kind': 'equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    masking_layers_from.append(output + '_sorted_chunked_mask_matching_hash')
  else:
    d[output + '_sorted_chunked_mask_valid_key_position'] = {
      'class': 'compare',
      'from': [output + '_sorted_chunked_stacked_keys_hashed'], 'value': hash_mask_value,
      'kind': 'not_equal'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
    masking_layers_from.append(output + '_sorted_chunked_mask_valid_key_position')
  if len(masking_layers_from) > 1:
    d[output + '_sorted_chunked_mask'] = {
      'class': 'compare', 'from': masking_layers_from,
      'kind': 'logical_and'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  else:
    d[output + '_sorted_chunked_mask'] = {
      'class': 'copy', 'from': masking_layers_from}  # [B,n,r,query-chunk,query-window,stacked-key-window]

  assert mask_current is not True

  # Compute chunked energy by comparing chunked queries and keys for each query chunk
  d[output + '_sorted_chunked_energy_unmasked'] = {
    'class': 'dot', 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': 'stag:query-window', 'var2': 'stag:stacked-key-window',
    'from': [output + '_sorted_chunked_queries', output + '_sorted_chunked_stacked_keys'],
    'debug': True}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_energy'] = {
    'class': 'switch', 'condition': output + '_sorted_chunked_mask',
    'true_from': output + '_sorted_chunked_energy_unmasked',
    'false_from': float('-inf')}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_energy_logsumexp'] = {
    'class': 'reduce', 'mode': 'logsumexp', 'axis': 'stag:stacked-key-window',
    'from': [output + '_sorted_chunked_energy']}  # [B,n,r,query-chunk,query-window]
  d[output + '_sorted_chunked_weights'] = {
    'class': 'eval', 'from': [output + '_sorted_chunked_energy', output + '_sorted_chunked_energy_logsumexp'],
    'eval': 'tf.exp(source(0) - source(1))'}  # [B,n,r,query-chunk,query-window,stacked-key-window]
  d[output + '_sorted_chunked_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_sorted_chunked_weights'],
    'dropout': dropout}  # [B,n,r,query-chunk,query-window,stacked-key-window]

  # Compute attention output of each round
  d[output + '_sorted_chunked_round_output'] = {
    'class': 'dot', 'red1': 'stag:stacked-key-window', 'red2': 'stag:stacked-key-window',
    'var1': 'stag:query-window', 'var2': 'static:-1', 'debug': True,
    'from': [
      output + '_sorted_chunked_weights_drop',
      output + '_sorted_chunked_stacked_values']}  # [B,n,r,query-chunk,query-window,F|d_v]

  # Undo chunking and undo sorting
  d[output + '_sorted_round_output'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk', 'stag:query-window'], 'keep_order': True,
    'from': [output + '_sorted_chunked_round_output']}  # [B,n,r,sorted-query-time=query-chunk*query-window,F|d_v]
  d[output + '_round_output'] = {
    'class': 'gather', 'from': [output + '_sorted_round_output'], 'axis': 'T',
    'position': output + '_queries_sort_indices'}  # [B,n,r,query-time,F|d_v]
  d[output + '_sorted_energy_logsumexp'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk', 'stag:query-window'], 'keep_order': True,
    'from': [output + '_sorted_chunked_energy_logsumexp']}  # [ B,n,r,sorted-query-time=query-chunk*query-window]
  d[output + '_energy_logsumexp'] = {
    'class': 'gather', 'from': [output + '_sorted_energy_logsumexp'], 'axis': 'T',
    'position': output + '_queries_sort_indices'}  # [B,n,r,query-time]

  # Combine the context vectors of the different rounds
  d[output + '_round_output_weights'] = {
    'class': 'softmax_over_spatial', 'axis': 'stag:att-rounds', 'use_time_mask': False, 'energy_factor': 1.0,
    'from': [output + '_energy_logsumexp']}  # [B,n,r,query-time]
  d[output + '_round_output_weighted'] = {
    'class': 'combine', 'kind': 'mul',
    'from': [output + '_round_output_weights', output + '_round_output']}  # [B,n,query-time,F|d_v]
  d[output + '_output'] = {
    'class': 'reduce', 'axis': 'stag:att-rounds', 'mode': 'sum',
    'from': [output + '_round_output_weighted']}  # [B,n,query-time,F|d_v]
  d[output + '_output_unnamed'] = {
    'class': 'name_axis', 'axis': 'stag:att-heads', 'description': None,
    'from': [output + '_output']}  # [B,query-time,F|n*d_v]
  d[output + '_att_all'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output_unnamed']}  # [B,query-time,F|n*d_v]

  if debug_print:
    for name in [output + n for n in [
      '_keys_hashed', '_queries_hashed',
      '_sorted_queries_orig_indices', '_sorted_keys_orig_indices',
      '_query_chunk_alignment']]:
      d[name + '_orig'] = d[name]
      d[name] = {'class': 'print', 'from': [name + '_orig']}
      # d[name + '_print'] = {
      #   'class': 'print', 'from': [name], 'is_output_layer': True}


def add_lsh_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None, *,
  num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0, num_hashes, chunk_size, chunks_before=None,
  chunks_after=None, ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  mask_current=True, small_mask_value=float(-10**5), mask_different_hashes=True, allow_duplicate_attention=False,
  debug_print=False):
  """
  Essentially this does (but for LSH attention)
    d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
      "total_key_dim": num_heads * key_dim,
      "n_out": num_heads * value_dim, "from": [input],
      "attention_left_only": left_only,
      "attention_dropout": dropout, "forward_weights_init": self.ff_init}
  But using multiple layers.

  :param dict[str,dict] d: the network dict to write into
  :param str input: input layer, of shape [B,query_axis?,F]
  :param str output: prefix of all layers generated. Output is written into output + '_att' layer.
    Will use the name output + '_...' for all internal layers here.
  :param bool inside_rec_layer: whether this is used inside a RecLayer, meaning that the time axis may or may not always
    exist.
  :param bool|None past_only: if set, will mask attention s.t. it cannot attend to the future.
    Must be set if used inside a RecLayer.
  :param str|None time_axis: name of the time axis
  :param int num_heads: number of attention heads
  :param int num_rounds: number of hashing rounds.
    Similar to attention heads but attend to same query/key/value sequence but with different hash matrix.
  :param int key_dim: feature dimension of keys and queries
  :param int value_dim: feature dimension of values
  :param int dropout: apply dropout to the attention weights
  :param int num_hashes: number of different attention hashes, must be an even number
  :param int chunk_size: window size within a single chunk
  :param int|None chunks_before: number of chunks we look into the past
  :param int|None chunks_after: number of chunks we look into the future
  :param str ff_init: initializer for the weight matrices, including the hash generator matrices
  :param bool mask_current: whether a query may attend to the key corresponding to the same position
  :param float|None mask_current_value: if mask_current, the attention energy if query=key is set to this.
    All other masked values are set to -inf, thus if mask_current_value is something low but higher than -inf, will
    attend to key=query exactly iff it is the only possible key to attend to
  :param bool small_mask_value: whether a query may only attend to keys with the same hash
  :param bool allow_duplicate_attention: whether to mask attention s.t. it only attends to each key once.
    Attending to a key twice can e.g. happen for multi-round attention,
    or if the (effective) chunk size is larger than the sequence length.
  :param bool debug_print: will print layers contents for debugging
  """
  if past_only is None:
    past_only = inside_rec_layer
  if time_axis is None:
    time_axis = 'stag:extern_data:classes' if inside_rec_layer else 'stag:extern_data:data'
  assert time_axis.startswith('stag:')
  assert not inside_rec_layer or past_only

  # Assume input [B,T|classes?,F|d_model]
  d[output + '_qv0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * (key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T|classes?,F|n*(d_k+d_v)]
  d[output + '_qv_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim + value_dim),
    'from': [output + '_qv0']}  # [B,T|classes?,n,F|d_k+d_v]
  d[output + '_qv'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_qv_unnamed']}  # [B,T|classes?,n,F|d_k+d_v]
  d[output + '_qv_split'] = {
    'class': 'split', 'axis': 'F', 'size_splits': (key_dim, value_dim),
    'from': [output + '_qv']}
  d[output + '_query'] = {
    'class': 'copy', 'from': [output + '_qv_split/0']}  # [B,T|classes?,n,F|d_k]
  d[output + '_key'] = {
    'class': 'eval', 'eval': normalize_eval, 'from': [output + '_query']}  # [B,T|classes?,n,F|d_k]
  d[output + '_value'] = {
    'class': 'copy', 'from': [output + '_qv_split/1']}  # [B,T|classes?,n,F|d_v]
  if inside_rec_layer:
    queries_input, keys_input, values_input = output + '_query_accum', output + '_key_accum', output + '_value_accum'
    for qkv in ('query', 'key', 'value'):
      d[output + '_%s_accum' % qkv] = {'class': 'cum_concat', 'from': [output + '_%s' % qkv], 'axis': time_axis}
    time_axis_ = 'stag:rec-history'
  else:
    queries_input, keys_input, values_input = output + '_query', output + '_key', output + '_value'
    time_axis_ = time_axis

  # this always computes attention for an entire sequence
  add_lsh_attention_layer(
    d, queries_input=queries_input, keys_input=keys_input, values_input=values_input,
    output=output, query_time_axis=time_axis_, key_time_axis=time_axis_,
    num_heads=num_heads, num_rounds=num_rounds, key_dim=key_dim, value_dim=value_dim, dropout=dropout,
    num_hashes=num_hashes, query_chunk_size=chunk_size, key_chunk_size=chunk_size,
    key_chunks_before=chunks_before, key_chunks_after=chunks_after, ff_init=ff_init,
    small_mask_value=small_mask_value, past_only=past_only, mask_current=mask_current,
    mask_different_hashes=mask_different_hashes, allow_duplicate_attention=allow_duplicate_attention,
    debug_print=debug_print)

  if inside_rec_layer:
    d[output + '_att'] = {'class': 'gather', 'from': [output + '_att_all'], 'position': ':i', 'axis': time_axis_}
  else:
    d[output + '_att'] = {'class': 'copy', 'from': [output + '_att_all']}