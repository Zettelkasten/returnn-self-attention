import tensorflow as tf

from returnn.tf.layers.basic import register_layer_class, _ConcatInputLayer
from returnn.tf.util.data import Data, DimensionTag


class NameAxisLayer(_ConcatInputLayer):
  """
  Adds a DimensionTag to an axis s.t. it will be unique.
  """
  layer_class = "name_axis"

  def __init__(self, axis, description, **kwargs):
    super(NameAxisLayer, self).__init__(**kwargs)

    # Maybe we still need to unbroadcast a size_placeholder
    if self.output.have_batch_axis():
      for i, dyn_size in self.output.size_placeholder.items():
        if len(dyn_size.shape) == 0 or dyn_size.shape[0] == 1:
          dim_tag = DimensionTag.get_tag_from_size_tensor(dyn_size)
          new_dyn_size = tf.broadcast_to(dyn_size, [tf.shape(self.output.placeholder)[self.output.batch_dim_axis]])
          dim_tag.set_tag_on_size_tensor(new_dyn_size)
          self.output.size_placeholder[i] = new_dyn_size

  @classmethod
  def get_out_data_from_opts(cls, name, axis, description, sources, **kwargs):
    """
    :param str name:
    :param str|int|list[str|int]|tuple[str|int] axis:
    :param str|None|list[str|None]|tuple[str|None] description:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    data = Data.get_common_data([s.output for s in sources])
    data = data.copy(name="%s_output" % name)

    if not isinstance(axis, (list, tuple)):
      axis = [axis]
    if not isinstance(description, (list, tuple)):
      description = [description]
    assert len(axis) == len(description)

    for ax, descr in zip(axis, description):
      if isinstance(ax, int):
        data = data.copy_as_batch_major()
      if isinstance(ax, str) and len(ax) >= 3 and ax[-2] == '+':
        ax_offset = int(ax[-1])
        ax = ax[:-2]
      else:
        ax_offset = 0
      ax = data.get_axis_from_description(ax, allow_int=True) + ax_offset
      ax_wo_batch = data.get_batch_axis_excluding_batch(ax)
      if descr is None:
        del data.size_placeholder[ax_wo_batch]
      else:
        if ax_wo_batch in data.size_placeholder:
          dyn_size = tf.identity(data.size_placeholder[ax_wo_batch])
        else:
          assert data.batch_shape[ax] is not None
          # this must actually be a [B]-tensor, but here it is not. we fix that later when we actually now the
          # placeholder (with the size we need to unbroadcast to)
          dyn_size = tf.constant(data.batch_shape[ax], shape=(1,))
        from returnn.tf.util.basic import DimensionTag
        tag = DimensionTag(
          description=descr,
          kind=DimensionTag.Types.Time)
        data.size_placeholder[ax_wo_batch] = dyn_size
        tag.set_tag_on_size_tensor(dyn_size)
    return data


register_layer_class(NameAxisLayer)


def maybe_gather(self, source, **kwargs):
  """
  Expects three inputs: (data, position, size_base).
  If data has no time dim axis, copy data.
  If has a time dim axis:
  1. copy size from size_base to data
  2. gather positions into data
  :return:
  """
  data = source(0, as_data=True, auto_convert=False)  # type: Data
  if data.have_time_axis():
    # copy size from source(2)
    size_base = source(2, as_data=True, auto_convert=False)  # type: Data
    data.size_placeholder[data.time_dim_axis_excluding_batch] = size_base.size_placeholder[size_base.time_dim_axis_excluding_batch]
    # apply positions from source(1)
    position = source(1, as_data=True, auto_convert=False)  # type: Data
    from returnn.tf.layers.base import InternalLayer
    from returnn.tf.layers.basic import GatherLayer
    input = InternalLayer(network=self.network, name="%s_input" % self.name, output=data)
    position = InternalLayer(network=self.network, name="%s_position" % self.name, output=position)
    kwargs = dict(network=self.network, name="%s_gather" % self.name, sources=[input], position=position, axis='T')
    layer = GatherLayer(**kwargs, output=GatherLayer.get_out_data_from_opts(**kwargs))
    return layer.output.placeholder
  else:
    # do nothing, but have to call source(1) and source(2)
    _, _ = source(1, auto_convert=False), source(2, auto_convert=False)
    return data.placeholder


def maybe_gather_template(network, name, sources, **kwargs):
  assert len(sources) == 3  # (data, position, size_base)
  data = sources[0].output  # type: Data
  if data.have_time_axis():
    # copy size from source(2)
    size_base = sources[2].output  # type: Data
    data.size_placeholder[data.time_dim_axis_excluding_batch] = size_base.size_placeholder[size_base.time_dim_axis_excluding_batch]

    # apply position using gather
    position = sources[1].output  # type: Data
    from returnn.tf.layers.base import InternalLayer
    from returnn.tf.layers.basic import GatherLayer
    input = InternalLayer(network=network, name="%s_input" % name, output=data)
    position = InternalLayer(network=network, name="%s_position" % name, output=position)
    kwargs = dict(network=network, name="%s_gather" % name, sources=[input], position=position, axis='T')
    out = GatherLayer.get_out_data_from_opts(**kwargs)
    # copy DimTag from input to be T|classes again
    out.size_placeholder[out.time_dim_axis_excluding_batch] = data.size_placeholder[data.time_dim_axis_excluding_batch]
    return out
  else:
    return data


def key_to_query_chunk(source, chunk_size, **kwargs):
  data = source(0, as_data=True, auto_convert=False)
  assert len(data.batch_shape) <= 1
  if data.have_time_axis():
    from returnn.tf.util.basic import get_shape
    # essentially a ceildiv
    num_chunks = tf.math.floordiv(get_shape(data.placeholder)[data.time_dim_axis] + chunk_size - 1, chunk_size)
    return tf.range(start=0, limit=num_chunks, dtype="int32")
  else:
    return data.placeholder // chunk_size

def key_to_query_chunk_template(name, sources, chunk_size, **kwargs):
  data = Data.get_common_data([s.output for s in sources])
  assert len(data.batch_shape) <= 1
  if data.have_time_axis():
    from returnn.tf.util.basic import DimensionTag
    # TODO: Not sure if this is correct. how many entries should this have?
    dyn_size = data.size_placeholder[data.get_batch_axis_excluding_batch(data.time_dim_axis)] // chunk_size
    tag = DimensionTag(
      description="query-chunk:%s" % name,
      kind=DimensionTag.Types.Time)
    data.size_placeholder[data.get_batch_axis_excluding_batch(data.time_dim_axis)] = dyn_size
    tag.set_tag_on_size_tensor(dyn_size)
  return data


def key_to_query_window(source, chunk_size, **kwargs):
  data = source(0, as_data=True, auto_convert=False)
  assert len(data.batch_shape) <= 1
  if data.have_time_axis():
    from returnn.tf.util.basic import get_shape
    return tf.range(start=0, limit=chunk_size)
  else:
    return data.placeholder % chunk_size


def key_to_query_window_template(name, sources, chunk_size, **kwargs):
  data = Data.get_common_data([s.output for s in sources])
  assert len(data.batch_shape) <= 1
  if data.have_time_axis():
    new_data = Data(
      name="%s_output" % name,
      batch_dim_axis=None,
      time_dim_axis=None,
      feature_dim_axis=None,
      shape=(chunk_size,),
      dim=None,
      dtype=data.dtype
    )
    dyn_size = tf.constant(chunk_size, shape=(1,))
    tag = DimensionTag(
      description="query-window:%s" % name,
      kind=DimensionTag.Types.Time)
    new_data.size_placeholder[0] = dyn_size
    tag.set_tag_on_size_tensor(dyn_size)
    return new_data
  else:
    return data


def add_lsh_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0, num_hashes=14, chunk_size=5, chunks_before=None, chunks_after=None,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  mask_current=True, mask_current_value=float(-10**5), mask_different_hashes=True,
  debug_print=False):
  """
  Essentially this does (but for LSH attention)
    d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
      "total_key_dim": num_heads * key_dim,
      "n_out": num_heads * value_dim, "from": [input],
      "attention_left_only": left_only,
      "attention_dropout": dropout, "forward_weights_init": self.ff_init}
  But using multiple layers that can be extended on
  :param d: the network dict to write into
  :param input: input layer, of shape [B,query_axis?,F]
  :param output: prefix of all layers generated. Output is written into output + '_att' layer.
    Will use the name output + '_...' for all internal layers here.
  :param bool inside_rec_layer: whether this is used inside a RecLayer, meaning that the time axis may or may not always
    exist.
  :param bool|None past_only: if set, will mask attention s.t. it cannot attend to the future.
    Must be set if used inside a RecLayer.
  :param str|None time_axis: name of the time axis
  :param int num_heads: number of attention heads
  :param int key_dim: feature dimension of keys and queries
  :param int value_dim: feature dimension of values
  :param int dropout: apply dropout to the attention weights
  :param int|None num_hashes: number of different attention hashes, must be an even number
  :param int chunk_size: window size within a single chunk
  :param int|None chunks_before: number of chunks we look into the past
  :param int|None chunks_after: number of chunks we look into the future
  :param str ff_init: initializer for the weight matrices, including the hash generator matrices
  :param bool mask_current: whether a query may attend to the key corresponding to the same position
  :param float|None mask_current_value: if mask_current, the attention energy if query=key is set to this.
    All other masked values are set to -inf, thus if mask_current_value is something low but higher than -inf, will
    attend to key=query exactly iff it is the only possible key to attend to
  :param bool mask_different_hashes: whether a query may only attend to keys with the same hash
  :param bool debug_print: will print layers contents for debugging
  """
  if past_only is None:
    past_only = inside_rec_layer
  if time_axis is None:
    time_axis = 'stag:extern_data:classes' if inside_rec_layer else 'stag:extern_data:data'
  assert time_axis.startswith('stag:')
  assert not inside_rec_layer or past_only
  if chunks_before is None:
    chunks_before = 1
  if chunks_after is None:
    chunks_after = 0 if past_only else 1
  assert chunks_before >= 0 and chunks_after >= 0
  chunk_stack_offsets_before = list(range(-chunks_before, 0))
  chunk_stack_offsets_after = list(range(1, chunks_after + 1))

  # Assume input [B,T|classes?,F|d_model]

  # Create (unnormalized) key/query, value
  d[output + '_kq0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * key_dim, 'forward_weights_init': ff_init}  # [B,T|classes?,F|n*d_k]
  d[output + '_kq_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim),
    'from': [output + '_kq0']}  # [B,T|classes?,n,F|d_k]
  d[output + '_kq'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_kq_unnamed']}  # [B,T|classes?,n,F|d_k]
  d[output + '_value0'] = {
    'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
    'n_out': num_heads * value_dim, 'forward_weights_init': ff_init}  # [B,T|classes?,F|n*d_v]
  d[output + '_value_unnamed'] = {
    'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, value_dim),
    'from': [output + '_value0']}  # [B,T|classes?,n,F|d_v]
  d[output + '_value'] = {
    'class': 'name_axis', 'axis': 'static:-2', 'description': 'att-heads',
    'from': [output + '_value_unnamed']}  # [B,T|classes?,n,F|d_v]

  # Mappings from key chunk dim to query chunk dim
  if inside_rec_layer:
    indices_from = ':i'
  else:
    d[output + '_indices'] = {
      'class': 'range_in_axis', 'from': [input], 'axis': time_axis, 'keepdims': False}  # [T|classes]
    indices_from = output + '_indices'
  d[output + '_key_to_query_chunk'] = {
    'class': 'eval', 'from': [indices_from], 'eval': key_to_query_chunk, 'out_type': key_to_query_chunk_template,
    'eval_locals': {'chunk_size': chunk_size}}  # [query_chunk_dim?] :: key_chunk_dim
  d[output + '_key_to_query_window'] = {
    'class': 'eval', 'from': [indices_from], 'eval': key_to_query_window, 'out_type': key_to_query_window_template,
    'eval_locals': {'chunk_size': chunk_size}}  # [query_window_dim?] :: key_window_dim

  # Hash the key/query
  assert num_hashes % 2 == 0
  hash_mask_value = 2**31-1
  assert hash_mask_value > num_hashes
  d[output + '_hash_gen_top_unnamed'] = {
    'class': 'variable', 'shape': (num_heads, key_dim, num_hashes // 2),
    'trainable': False, 'init': ff_init, 'add_batch_axis': True}  # [B,n,d_k,F|d_h/2]
  d[output + '_hash_gen_top'] = {
    'class': 'name_axis', 'axis': 'static:0', 'description': 'att-heads',
    'from': [output + '_hash_gen_top_unnamed']}  # [B,n,d_k,F|d_h/2]
  d[output + '_hash_gen_bottom'] = {
    'class': 'eval', 'eval': '-source(0)',
    'from': [output + '_hash_gen_top']}  # [B,n,d_k,F|d_h/2]
  d[output + '_hash_gen'] = {
    'class': 'copy',
    'from': [output + '_hash_gen_top', output + '_hash_gen_bottom']}  # [B,n,d_k,F|d_h]
  d[output + '_kq_hash_linear'] = {
    'class': 'dot', 'from': [output + '_hash_gen', output + '_kq'],
    'red1': 'static:-2', 'red2': 'static:-1', 'var1': 'static:-1',
    'var2': time_axis + '?', 'add_var2_if_empty': False}  # [B,T|classes?,n,F|d_h]
  d[output + '_kq_hash_sparse'] = {
    'class': 'reduce', 'mode': 'argmax', 'axes': 'static:-1',
    'from': [output + '_kq_hash_linear']}  # [B,T|classes?,n] :: d_h
  d[output + '_kq_hash'] = {
    'class': 'reinterpret_data', 'from': [output + '_kq_hash_sparse'],
    'set_sparse': False, 'set_axes': {'F': None}}  # [B,T|classes?,n] :: d_h

  # Accumulate all past hashes, and create chunks
  d[output + '_kq_accum_hash_unmasked'] = {
    'class': 'cum_concat', 'from': [output + '_kq_hash']}  # [B,T|rec-history,n] :: d_h
  d[output + '_kq_accum_hash'] = {
    'class': 'seq_len_mask', 'from': [output + '_kq_accum_hash_unmasked'],
    'axis': 'stag:rec-history',
    'mask_value': hash_mask_value}  # [B,T|rec-history,n] :: d_h
  d[output + '_kq_accum_hash_chunked_feature'] = {
    'class': 'split_dims', 'from': [output + '_kq_accum_hash'],
    'axis': 'stag:rec-history', 'dims': [-1, chunk_size],
    'pad_value': hash_mask_value}  # [key_chunk_dim,key_window_dim,B,n] :: d_h
  d[output + '_kq_accum_hash_chunked_unnamed'] = {
    'class': 'reinterpret_data', 'from': [output + '_kq_accum_hash_chunked_feature'],
    'set_axes': {'F': None}}  # [key_chunk_dim,key_window_dim,B,n] :: d_h
  d[output + '_kq_accum_hash_chunked'] = {
    'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
    'from': [output + '_kq_accum_hash_chunked_unnamed']}  # [key_chunk_dim,key_window_dim,B,n] :: d_h
  d[output + '_key_accum_hash_chunked_single'] = {
    'class': 'copy',
    'from': [output + '_kq_accum_hash_chunked']}  # [key_chunk_dim,key_window_dim,B,n] :: d_h
  for stack_offset in chunk_stack_offsets_before + chunk_stack_offsets_after:
    d[output + '_key_accum_hash_chunked_offset%s' % stack_offset] = {
      'class': 'eval',
      'from': [output + '_key_accum_hash_chunked_single'],
      'eval': 'tf.roll(source(0), shift=-chunk_offset, axis=source(0, as_data=True).get_axis_from_description("stag:key-chunk"))',
      'eval_locals': {'chunk_offset': stack_offset}}  # [key_chunk_dim,key_window_dim,B,n] :: d_h
  d[output + '_key_accum_hash_chunked_stacked_split'] = {
    'class': 'stack',
    'from': (
        [output + '_key_accum_hash_chunked_offset%s' % i for i in chunk_stack_offsets_before]
        + [output + '_key_accum_hash_chunked_single']
        + [output + '_key_accum_hash_chunked_offset%s' % i for i in chunk_stack_offsets_after])}  # [key_chunk_dim,key_window_dim,chunk_stack_dim,B,n] :: d_h
  d[output + '_key_accum_hash_chunked_stacked_unnamed'] = {
    'class': 'merge_dims', 'axes': ['stag:key-window', 'spatial:-1'],
    'from': [output + '_key_accum_hash_chunked_stacked_split']}  # [key_chunk_dim,2*key_window_dim,B,n] :: d_h
  d[output + '_key_accum_hash_chunked_stacked'] = {
    'class': 'name_axis', 'axis': 'T+1', 'description': 'key-stacked-window',
    'from': [output + '_key_accum_hash_chunked_stacked_unnamed']}  # [key_chunk_dim,2*key_window_dim,B,n] :: d_h
  d[output + '_key_accum_hash_chunked'] = {
    'class': 'gather', 'from': [output + '_key_accum_hash_chunked_stacked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,2*key_window_dim,B,n] :: d_h
  d[output + '_query_hash_chunked_all'] = {
    'class': 'gather', 'from': [output + '_kq_accum_hash_chunked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,key_window_dim,B,n] :: d_h
  d[output + '_query_hash_chunked'] = {
    'class': 'gather', 'from': [output + '_query_hash_chunked_all'],
    'axis': 'stag:key-window',
    'position': output + '_key_to_query_window'}  # [query_chunk_dim?,query_window_dim?,B,n] :: d_h

  # Sort the hashes
  d[output + '_kq_accum_sort_to_orig'] = {
    'class': 'eval',
    'eval': 'tf.argsort(source(0), axis=source(0, as_data=True).get_axis_from_description("stag:rec-history"), direction="ASCENDING", stable=True)',
    'from': [output + '_kq_accum_hash']}  # [B,T|rec-history,n] :: T|rec-history
  d[output + '_kq_accum_sort_to_orig_chunked_feature'] = {
    'class': 'split_dims', 'from': [output + '_kq_accum_sort_to_orig'],
    'axis': 'stag:rec-history', 'dims': [-1, chunk_size]}  # [key_chunk_dim,key_window_dim,B,n] :: T|rec-history
  d[output + '_kq_accum_sort_to_orig_chunked_unnamed'] = {
    'class': 'reinterpret_data', 'from': [output + '_kq_accum_sort_to_orig_chunked_feature'],
    'set_axes': {'F': None}}  # [key_chunk_dim,key_window_dim,B,n] :: T|rec-history
  d[output + '_kq_accum_sort_to_orig_chunked'] = {
    'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
    'from': [output + '_kq_accum_sort_to_orig_chunked_unnamed']}  # [key_chunk_dim,key_window_dim,B,n] :: T|rec-history
  d[output + '_key_accum_sort_to_orig_chunked_single'] = {
    'class': 'copy',
    'from': [output + '_kq_accum_sort_to_orig_chunked']}  # [key_chunk_dim,key_window_dim,B,n] :: T|rec-history
  for stack_offset in chunk_stack_offsets_before + chunk_stack_offsets_after:
    d[output + '_key_accum_sort_to_orig_chunked_offset%s' % stack_offset] = {
      'class': 'eval',
      'from': [output + '_key_accum_sort_to_orig_chunked_single'],
      'eval': 'tf.roll(source(0), shift=-chunk_offset, axis=source(0, as_data=True).get_axis_from_description("stag:key-chunk"))',
      'eval_locals': {'chunk_offset': stack_offset}}  # [key_chunk_dim,key_window_dim,B,n] :: T|rec-history
  d[output + '_key_accum_sort_to_orig_chunked_stacked_split'] = {
    'class': 'stack',
    'from': (
        [output + '_key_accum_sort_to_orig_chunked_offset%s' % i for i in chunk_stack_offsets_before]
        + [output + '_key_accum_sort_to_orig_chunked_single']
        + [output + '_key_accum_sort_to_orig_chunked_offset%s' % i for i in chunk_stack_offsets_after])}  # [key_chunk_dim,key_window_dim,chunk_stack_dim,B,n] :: T|rec-history
  d[output + '_key_accum_sort_to_orig_chunked_stacked_unnamed'] = {
    'class': 'merge_dims', 'axes': ['stag:key-window', 'spatial:-1'],
    'from': [output + '_key_accum_sort_to_orig_chunked_stacked_split']}  # [key_chunk_dim,2*key_window_dim,B,n] :: T|rec-history
  d[output + '_key_accum_sort_to_orig_chunked_stacked'] = {
    'class': 'name_axis', 'axis': ['T+1'], 'description': ['key-stacked-window'],
    'from': [output + '_key_accum_sort_to_orig_chunked_stacked_unnamed']}  # [key_chunk_dim,2*key_window_dim,B,n] :: T|rec-history
  d[output + '_key_accum_sort_to_orig_chunked'] = {
    'class': 'gather', 'from': [output + '_key_accum_sort_to_orig_chunked_stacked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,2*key_window_dim,B,n] :: T|rec-history
  d[output + '_query_sort_to_orig_chunked_all'] = {
    'class': 'gather', 'from': [output + '_kq_accum_sort_to_orig_chunked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,key_window_dim,B,n] :: T|rec-history
  d[output + '_query_sort_to_orig_chunked'] = {
    'class': 'gather', 'from': [output + '_query_sort_to_orig_chunked_all'],
    'axis': 'stag:key-window',
    'position': output + '_key_to_query_window'}  # [query_chunk_dim?,query_window_dim?,B,n] :: T|rec-history

  # Invert permutation to undo sorting later
  d[output + '_kq_accum_orig_indices'] = {
    'class': 'range_in_axis', 'from': [output + '_kq_accum_hash'], 'axis': 'stag:rec-history',
    'keepdims': False}  # [T|rec-history] :: T|rec-history
  d[output + '_kq_accum_orig_to_sort'] = {
    'class': 'scatter_nd', 'from': [output + '_kq_accum_orig_indices'],
    'position': output + '_kq_accum_sort_to_orig', 'position_axis': 'stag:rec-history',
    'output_dim_via_time_from': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n] :: T|rec-history

  # Accumulate all past keys/queries
  d[output + '_kq_accum_unsorted'] = {
    'class': 'cum_concat', 'from': [output + '_kq']}  # [B,T|rec-history,n,F|d_k]
  d[output + '_kq_accum'] = {
    'class': 'gather', 'from': [output + '_kq_accum_unsorted'],
    'axis': 'stag:rec-history',
    'position': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,F|d_k]
  d[output + '_kq_accum_chunked_unnamed'] = {
    'class': 'split_dims', 'from': [output + '_kq_accum'],
    'axis': 'stag:rec-history',
    'dims': [-1, chunk_size]}  # [key_chunk_dim,key_window_dim,B,n,F|d_k]
  d[output + '_kq_accum_chunked'] = {
    'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
    'from': [output + '_kq_accum_chunked_unnamed']}  # [key_chunk_dim,key_window_dim,B,n,F|d_k]

  # Compute attention keys
  d[output + '_key_accum_chunked_single'] = {
    'class': 'eval',
    'eval': 'tf.math.divide_no_nan(source(0), tf.norm(source(0), axis=source(0, as_data=True).feature_dim_axis, keepdims=True))',
    'from': [output + '_kq_accum_chunked']}  # [key_chunk_dim,key_window_dim,B,n,F|d_k]
  for stack_offset in chunk_stack_offsets_before + chunk_stack_offsets_after:
    d[output + '_key_accum_chunked_offset%s' % stack_offset] = {
      'class': 'eval',
      'from': [output + '_key_accum_chunked_single'],
      'eval': 'tf.roll(source(0), shift=-chunk_offset, axis=source(0, as_data=True).get_axis_from_description("stag:key-chunk"))',
      'eval_locals': {'chunk_offset': stack_offset}}  # [key_chunk_dim,key_window_dim,B,n,F|d_k]
  d[output + '_key_accum_chunked_stacked_split'] = {
    'class': 'stack',
    'from': (
        [output + '_key_accum_chunked_offset%s' % i for i in chunk_stack_offsets_before]
        + [output + '_key_accum_chunked_single']
        + [output + '_key_accum_chunked_offset%s' % i for i in chunk_stack_offsets_after])}  # [key_chunk_dim,key_window_dim,chunk_stack_dim,B,n,F|d_k]
  d[output + '_key_accum_chunked_stacked_unnamed'] = {
    'class': 'merge_dims', 'axes': ['stag:key-window', 'spatial:-1'],
    'from': [output + '_key_accum_chunked_stacked_split']}  # [key_chunk_dim,2*key_window_dim,B,n,F|d_k]
  d[output + '_key_accum_chunked_stacked'] = {
    'class': 'name_axis', 'axis': ['T+1'], 'description': ['key-stacked-window'],
    'from': [output + '_key_accum_chunked_stacked_unnamed']}  # [key_chunk_dim,2*key_window_dim,B,n,F|d_k]
  d[output + '_key_accum_chunked'] = {
    'class': 'gather', 'from': [output + '_key_accum_chunked_stacked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,2*key_window_dim,B,n,F|d_k]

  # Compute attention queries
  d[output + '_query_chunked_all'] = {
    'class': 'gather', 'from': [output + '_kq_accum_chunked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,key_window_dim,B,n,F|d_k]
  d[output + '_query_chunked'] = {
    'class': 'gather', 'from': [output + '_query_chunked_all'],
    'axis': 'stag:key-window',
    'position': output + '_key_to_query_window'}  # [query_chunk_dim?,query_window_dim?,B,n,F|d_k]

  # Compute energy mask
  masking_layers_from = []
  if past_only:
    d[output + '_energy_chunked_mask_past_only'] = {
      'class': 'compare',
      'from': [output + '_query_sort_to_orig_chunked', output + '_key_accum_sort_to_orig_chunked'],
      'kind': 'greater_equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
    masking_layers_from.append(output + '_energy_chunked_mask_past_only')
  if mask_different_hashes:
    # Masking valid positions is not necessary in this case as invalid positions will be masked with a special
    # hash value
    d[output + '_energy_chunked_mask_matching_hash'] = {
      'class': 'compare',
      'from': [output + '_query_hash_chunked', output + '_key_accum_hash_chunked'],
      'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
    masking_layers_from.append(output + '_energy_chunked_mask_matching_hash')
  else:
    d[output + '_energy_chunked_mask_valid_key_position'] = {
      'class': 'compare',
      'from': [output + '_key_accum_hash_chunked'], 'value': hash_mask_value,
      'kind': 'not_equal'}  # [B,query_chunk_dim?,2*key_window_dim,n]
    masking_layers_from.append(output + '_energy_chunked_mask_valid_key_position')
  if len(masking_layers_from) > 1:
    d[output + '_energy_chunked_mask'] = {
      'class': 'compare',
      'from': masking_layers_from,
      'kind': 'logical_and'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  else:
    d[output + '_energy_chunked_mask'] = {
      'class': 'copy', 'from': masking_layers_from}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]

  # Compute energy small mask (i.e. the entries that will be set to mask_current_value)
  small_masking_layers_from = [output + '_energy_chunked_mask_small_invalid_query_position']
  # We never want the attention weights to be NaN for any query (even for unmasked queries),
  # and thus need to have at least one masked key for every query.
  # Otherwise, the gradients will be NaN.
  # We ensure this by masking all energies with a small (finite) number.
  d[output + '_energy_chunked_mask_small_invalid_query_position'] = {
    'class': 'compare',
    'from': [output + '_query_hash_chunked'], 'value': hash_mask_value,
    'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,n]
  if mask_current:
    d[output + '_energy_chunked_mask_small_current'] = {
      'class': 'compare',
      'from': [output + '_query_sort_to_orig_chunked', output + '_key_accum_sort_to_orig_chunked'],
      'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
    small_masking_layers_from.append(output + '_energy_chunked_mask_small_current')
  if len(small_masking_layers_from) > 1:
    d[output + '_energy_chunked_mask_small'] = {
      'class': 'compare',
      'from': small_masking_layers_from,
      'kind': 'logical_or'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  else:
    d[output + '_energy_chunked_mask_small'] = {
      'class': 'copy', 'from': small_masking_layers_from}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]

  # Compute energy
  d[output + '_energy_chunked_feature'] = {
    'class': 'dot', 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': 'stag:query-window?', 'var2': 'stag:key-stacked-window',
    'from': [output + '_query_chunked', output + '_key_accum_chunked'],
    'debug': True}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  d[output + '_energy_chunked_unmasked'] = {
    'class': 'reinterpret_data', 'from': [output + '_energy_chunked_feature'],
    'set_axes': {'F': None}}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  d[output + '_energy_chunked_not_small'] = {
    'class': 'switch', 'condition': output + '_energy_chunked_mask',
    'true_from': output + '_energy_chunked_unmasked',
    'false_from': float('-inf')}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  d[output + '_energy_chunked'] = {
    'class': 'switch', 'condition': output + '_energy_chunked_mask_small',
    'true_from': mask_current_value,
    'false_from': output + '_energy_chunked_not_small'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  d[output + '_weights_chunked'] = {
    'class': 'softmax_over_spatial', 'axis': 'stag:key-stacked-window',
    'use_time_mask': False, 'energy_factor': key_dim ** -0.5,
    'from': [output + '_energy_chunked']}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]
  d[output + '_weights_chunked_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights_chunked'],
    'dropout': dropout}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n]

  # Compute attention values
  d[output + '_value_accum_unsorted'] = {
    'class': 'cum_concat', 'from': [output + '_value']}  # [B,T|rec-history,n,F|d_v]
  d[output + '_value_accum'] = {
    'class': 'gather', 'from': [output + '_value_accum_unsorted'],
    'axis': 'stag:rec-history',
    'position': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,F|d_v]
  d[output + '_value_accum_chunked_unnamed'] = {
    'class': 'split_dims', 'from': [output + '_value_accum'],
    'axis': 'stag:rec-history',
    'dims': [-1, chunk_size]}  # [key_chunk_dim,key_window_dim,B,n,F|d_v]
  d[output + '_value_accum_chunked_single'] = {
    'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
    'from': [output + '_value_accum_chunked_unnamed']}  # [key_chunk_dim,key_window_dim,B,n,F|d_v]
  for stack_offset in chunk_stack_offsets_before + chunk_stack_offsets_after:
    d[output + '_value_accum_chunked_offset%s' % stack_offset] = {
      'class': 'eval',
      'from': [output + '_value_accum_chunked_single'],
      'eval': 'tf.roll(source(0), shift=-chunk_offset, axis=source(0, as_data=True).get_axis_from_description("stag:key-chunk"))',
      'eval_locals': {'chunk_offset': stack_offset}}  # [key_chunk_dim,key_window_dim,B,n,F|d_v]
  d[output + '_value_accum_chunked_stacked_split'] = {
    'class': 'stack',
    'from': (
        [output + '_value_accum_chunked_offset%s' % i for i in chunk_stack_offsets_before]
        + [output + '_value_accum_chunked_single']
        + [output + '_value_accum_chunked_offset%s' % i for i in chunk_stack_offsets_after])}  # [key_chunk_dim,key_window_dim,chunk_stack_dim,B,n,F|d_v]
  d[output + '_value_accum_chunked_stacked_unnamed'] = {
    'class': 'merge_dims', 'axes': ['stag:key-window', 'spatial:-1'],
    'from': [output + '_value_accum_chunked_stacked_split']}  # [key_chunk_dim,2*key_window_dim,B,n,F|d_v]
  d[output + '_value_accum_chunked_stacked'] = {
    'class': 'name_axis', 'axis': ['T+1'], 'description': ['key-stacked-window'],
    'from': [output + '_value_accum_chunked_stacked_unnamed']}  # [key_chunk_dim,2*key_window_dim,B,n,F|d_v]
  d[output + '_value_accum_chunked'] = {
    'class': 'gather', 'from': [output + '_value_accum_chunked_stacked'],
    'axis': 'stag:key-chunk',
    'position': output + '_key_to_query_chunk'}  # [query_chunk_dim?,2*key_window_dim,B,n,F|d_v]

  # Compute the outputted weighted sum (i.e. the context vector)
  d[output + '_output_chunked'] = {
    'class': 'dot', 'red1': 'stag:key-stacked-window', 'red2': 'stag:key-stacked-window',
    'var1': 'stag:query-window?', 'var2': 'static:-1', 'debug': True,
    'from': [output + '_weights_chunked_drop', output + '_value_accum_chunked']}  # [B,query_chunk_dim?,query_window_dim?,n,F|d_v]
  d[output + '_output_sorted'] = {
    'class': 'merge_dims',
    'axes': ['stag:query-chunk?', 'stag:query-window?'],
    'from': [output + '_output_chunked']}  # [B,query_chunk_dim?*query_window_dim?,n,F|d_v]
  d[output + '_output'] = {
    'class': 'eval', 'from': [output + '_output_sorted', output + '_kq_accum_orig_to_sort', output + '_value'],
    'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,T|classes?,n,F|d_k]
  d[output + '_output_unnamed'] = {
    'class': 'name_axis', 'axis': 'stag:att-heads', 'description': None,
    'from': [output + '_output']}  # [B,T|classes?,F|n*d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output_unnamed']}  # [B,T|classes?,F|n*d_v]

  if debug_print:
    for name in [output + n for n in [
      '_kq', '_value', '_key_to_query_chunk', '_key_to_query_window',
      '_kq_hash', '_kq_accum_hash', '_kq_accum_hash_chunked',
      '_key_accum_hash_chunked', '_query_hash_chunked', '_kq_accum_sort_to_orig', '_kq_accum_orig_to_sort',
      '_kq_accum_unsorted', '_kq_accum', '_kq_accum_chunked', '_key_accum_chunked', '_query_chunked',
      '_energy_chunked_unmasked', '_energy_chunked_not_small', '_query_sort_to_orig_chunked', '_key_accum_sort_to_orig_chunked',
      '_energy_chunked_mask', '_energy_chunked_mask_small',
      '_energy_chunked', '_weights_chunked', '_value_accum_chunked',
       '_output_chunked', '_output_sorted', '_output', '_att'
    ]] + masking_layers_from + small_masking_layers_from:
      d[name + '_orig'] = d[name]
      d[name] = {'class': 'print', 'from': [name + '_orig']}
      #d[name + '_print'] = {
      #  'class': 'print', 'from': [name], 'is_output_layer': True}


def add_vanilla_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  share_key_query=False,
  mask_current=False, mask_current_value=float(-10**5)):
  """
  Essentially this does
    d[output + '_att'] = {"class": "self_attention", "num_heads": num_heads,
      "total_key_dim": num_heads * key_dim,
      "n_out": num_heads * value_dim, "from": [input],
      "attention_left_only": past_only,
      "attention_dropout": dropout, "forward_weights_init": self.ff_init}
  But using multiple layers that can be extended on
  """
  if past_only is None:
    past_only = inside_rec_layer
  if time_axis is None:
    time_axis = 'stag:extern_data:classes' if inside_rec_layer else 'stag:extern_data:data'
  assert time_axis.startswith('stag:')
  assert not inside_rec_layer or past_only

  # Create (non-accumulated) query, key and value
  if not share_key_query:
    d[output + '_qkv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (2 * key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T?,F|n*(2d_k+d_v)]
    d[output + '_qkv'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, 2 * key_dim + value_dim),
      'from': [output + '_qkv0']}  # [B,T?,n,F|2d_k+d_v]
    d[output + '_qkv_split'] = {
      'class': 'split', 'axis': 'F', 'size_splits': (key_dim, key_dim, value_dim),
      'from': [output + '_qkv']}
    d[output + '_query'] = {
      'class': 'copy', 'from': [output + '_qkv_split/0']}  # [B,T?,n,F|d_k]
    d[output + '_key'] = {
      'class': 'copy', 'from': [output + '_qkv_split/1']}  # [B,T?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qkv_split/2']}  # [B,T?,n,F|d_v]
  else:  # share_key_query
    assert False, 'not implemented yet'

  # Accumulate keys/values
  d[output + '_key_accum'] = {
    'class': 'cum_concat', 'from': [output + '_key']}  # [B,T|rec-history,n,F|d_k]
  d[output + '_value_accum'] = {
    'class': 'cum_concat', 'from': [output + '_value']}  # [B,T|rec-history,n,F|d_v]

  # Calculate the energies
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', output + '_key_accum'],
    'red1': 'static:-1', 'red2': 'static:-1', 'var1': time_axis + '?', 'var2': 'stag:rec-history'}  # [B,n,T?,T|rec-history]
  if past_only:
    d[output + '_energy_unmasked'] = d[output + '_energy']
    if inside_rec_layer:
      query_indices_from = ':i'
    else:
      d[output + '_query_indices'] = {'class': 'range_in_axis', 'from': [input], 'axis': time_axis, 'keepdims': False}  # [T]
      query_indices_from = output + '_query_indices'
    d[output + '_key_accum_indices'] = {
      'class': 'range_in_axis', 'from': [output + '_key_accum'], 'axis': 'stag:rec-history', 'keepdims': False}  # [T|rec-history]
    d[output + '_energy_mask'] = {
      'class': 'compare', 'kind': 'greater_equal', 'from': [query_indices_from, output + '_key_accum_indices']}
    d[output + '_energy'] = {
      'class': 'switch', 'true_from': output + '_energy_unmasked', 'false_from': float('-inf'),
      'condition': output + '_energy_mask'}  # [B,n,T?,T|rec-history]
    assert not mask_current, 'not implemented yet'
  # If past_only=True, do not apply a time mask here, as we apply our own masking using energy_mask.
  # If we would apply additional masking here, we would mask away all keys for queries that are unmasked, giving
  # attention weights NaN for these queries. Even though these are masked away later in the forward pass, the gradient
  # can still become NaN.
  # If past_only=False, do apply the normal time mask.
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': 'stag:rec-history',
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': not past_only}  # [B,n,T?,T|rec-history]
  d[output + '_output'] = {
    'class': 'dot', 'from': [output + '_weights', output + '_value_accum'],
    'red1': 'stag:rec-history', 'red2': 'stag:rec-history', 'var1': time_axis + '?', 'var2': 'static:-1'}  # [B,n,T?,F|d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output']}  # [B,T?,F|n*d_v]
