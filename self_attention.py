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
    # As the output does not necessarily have a batch dim, but the size_placeholder still needs a batch dim,
    # we use the global batch dim here.
    from returnn.tf.layers.base import LayerBase
    batch_dim = LayerBase.get_recent_layer().get_batch_info().dim
    for i, dyn_size in self.output.size_placeholder.items():
      if len(dyn_size.shape) == 0 or dyn_size.shape[0] == 1:
        dim_tag = DimensionTag.get_tag_from_size_tensor(dyn_size)
        new_dyn_size = tf.broadcast_to(dyn_size, [batch_dim])
        dim_tag.set_tag_on_size_tensor(new_dyn_size)
        dim_tag.dyn_size = new_dyn_size  # override this explicitly: dim_tag.set_tag_on_size_tensor does not reset it.
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
      if isinstance(ax, str) and '|' in ax:
        possible_axes = ax.split('|')
        found_ax = None
        for possible_ax in possible_axes:
          try:
            found_ax = data.get_axis_from_description(possible_ax)
            break
          except:
            continue
        assert found_ax is not None, '%r: axis %r not found in %r' % (cls, ax, data)
        ax = found_ax
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
  if data.have_time_axis():
    from returnn.tf.util.basic import get_shape
    # ceildiv(dyn_size, chunk_size)
    num_chunks = tf.math.floordiv(get_shape(data.placeholder)[data.time_dim_axis] + chunk_size - 1, chunk_size)
    return tf.range(start=0, limit=num_chunks, dtype="int32")
  else:
    return data.placeholder // chunk_size


def key_to_query_chunk_template(name, sources, chunk_size, **kwargs):
  data = Data.get_common_data([s.output for s in sources])
  if data.have_time_axis():
    new_data = Data('%s_output' % name, batch_dim_axis=None, time_dim_axis=0, feature_dim_axis=None, dtype=data.dtype)
    from returnn.tf.util.basic import DimensionTag, get_shape
    # ceildiv(dyn_size, chunk_size)
    dyn_size = tf.math.floordiv(data.size_placeholder[data.time_dim_axis_excluding_batch] + chunk_size - 1, chunk_size)
    tag = DimensionTag(
      description="query-chunk:%s" % name,
      kind=DimensionTag.Types.Time)
    new_data.size_placeholder[new_data.time_dim_axis_excluding_batch] = dyn_size
    tag.set_tag_on_size_tensor(dyn_size)
    return new_data
  else:
    return data.copy('%s_output' % name)


def key_to_query_window(source, chunk_size, **kwargs):
  data = source(0, as_data=True, auto_convert=False)
  if data.have_time_axis():
    return tf.range(start=0, limit=chunk_size)
  else:
    return data.placeholder % chunk_size


def key_to_query_window_template(name, sources, chunk_size, **kwargs):
  data = Data.get_common_data([s.output for s in sources])
  if data.have_time_axis():
    new_data = Data('%s_output' % name, batch_dim_axis=None, time_dim_axis=0, feature_dim_axis=None, dtype=data.dtype)
    from returnn.tf.util.basic import get_shape
    dyn_size = tf.fill(get_shape(data.size_placeholder[data.time_dim_axis_excluding_batch]), chunk_size)
    tag = DimensionTag(
      description="query-window:%s" % name,
      kind=DimensionTag.Types.Time)
    new_data.size_placeholder[new_data.time_dim_axis_excluding_batch] = dyn_size
    tag.set_tag_on_size_tensor(dyn_size)
    return new_data
  else:
    return data.copy('%s_output' % name)


normalize_eval = 'tf.math.divide_no_nan(source(0), tf.norm(source(0), axis=source(0, as_data=True).feature_dim_axis, ' \
                 'keepdims=True))'

argsort_eval = 'tf.argsort(source(0), axis=source(0, as_data=True).get_axis_from_description("%s"), ' \
               'direction="ASCENDING", stable=True)'

clip_eval = 'tf.where(source(0) == mask_value, 0, source(0))'


def make_lsh_hash_gen(d, output, key_dim, num_hashes, num_heads, num_rounds,
                      hash_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0):
  """
  :param dict[str,dict] d: the network dict to write into
  :param str output: prefix of all layers generated. Output is written into output + '_hash_gen' layer.
  :param int key_dim:
  :param int num_hashes:
  :param int num_heads:
  :param int num_rounds:
  :param str hash_init: initializer for the hash generator matrix
  """
  assert num_hashes % 2 == 0
  d[output + '_top_unnamed'] = {
    'class': 'variable', 'shape': (num_heads, num_rounds, key_dim, num_hashes // 2),
    'trainable': False, 'init': hash_init, 'add_batch_axis': True}  # [B,n,r,d_k,F|d_h/2]
  d[output + '_top'] = {
    'class': 'name_axis', 'axis': ['static:0', 'static:1'], 'description': ['att-heads', 'att-rounds'],
    'from': [output + '_top_unnamed']}  # [B,n,r,d_k,F|d_h/2]
  d[output + '_bottom'] = {
    'class': 'eval', 'eval': '-source(0)',
    'from': [output + '_top']}  # [B,n,r,d_k,F|d_h/2]
  d[output] = {
    'class': 'copy',
    'from': [output + '_top', output + '_bottom']}  # [B,n,r,d_k,F|d_h]


def apply_lsh_hash_gen(d, input, hash_gen_input, output, num_hashes, time_axis, hash_mask_value=2 ** 31 - 1,
                       hash_dropin=0.0):
  """
  :param dict[str,dict] d:
  :param str input:
  :param str hash_gen_input:
  :param str output:
  :param int num_hashes:
  :param str time_axis:
  :param int|None hash_mask_value: or None if you do not want masking
  :param float hash_dropin:
  """
  d[output + '_linear'] = {
    'class': 'dot', 'from': [hash_gen_input, input], 'debug': True,
    'red1': 'static:-2', 'red2': 'F', 'var1': ['stag:att-rounds', 'static:-1'],
    'var2': time_axis + '?', 'add_var2_if_empty': False}  # [B,T|classes?,n,r,F|d_h]
  d[output + '_sparse'] = {
    'class': 'reduce', 'mode': 'argmax', 'axes': 'static:-1',
    'from': [output + '_linear']}  # [B,T|classes?,n,r] :: d_h
  d[output + '_actual'] = {
    'class': 'reinterpret_data', 'from': [output + '_sparse'],
    'set_sparse': False, 'set_axes': {'F': None}}  # [B,T|classes?,n,r] :: d_h
  # DropoutLayer does not support inputs that are not of type float.
  d[output + '_dropin_decision_ones'] = {
    'class': 'eval', 'from': [output + '_actual'], 'eval': 'tf.ones_like(source(0), dtype="float32")',
    'out_type': {'dtype': 'float32'}}  # [B,T|classes?,n,r] :: 1.0
  d[output + '_dropin_decision_float'] = {
    'class': 'dropout', 'dropout': hash_dropin, 'dropout_noise_shape': {'B': -1, 'except_time': -1, 'T': 1},
    'from': [output + '_dropin_decision_ones']}  # [B,T|classes?,n,r] :: 0.0/1.0
  d[output + '_dropin_decision'] = {
    'class': 'compare', 'from': [output + '_dropin_decision_float'], 'kind': 'greater',
    'value': 0.5}  # [B,T|classes?,n,r] :: False/True
  d[output + '_dropin_hashes'] = {
    'class': 'eval',
    'eval': 'tf.random.uniform(tf.shape(source(0)), minval=0, maxval=%s, dtype="int32")' % num_hashes,
    'from': [output + '_actual'], 'out_type': {'dtype': 'int32'}}  # [B,T|classes?,n,r] :: d_h
  d[output + '_unmasked'] = {
    'class': 'switch', 'condition': output + '_dropin_decision', 'true_from': output + '_actual',
    'false_from': output + '_dropin_hashes'}  # [B,T|classes?,n,r] :: d_h
  if hash_mask_value is not None:
    d[output] = {
      'class': 'seq_len_mask', 'from': [output + '_unmasked'], 'axis': time_axis,
      'mask_value': hash_mask_value}  # [B,T|classes?,n,r] :: d_h
  else:
    d[output] = {'class': 'copy', 'from': [output + '_unmasked']}  # [B,T|classes?,n,r] :: d_h


def legacy_add_lsh_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None,
  num_heads=8, num_rounds=1, key_dim=64, value_dim=64, dropout=0.0, num_hashes=14, chunk_size=5, chunks_before=None,
  chunks_after=None, ff_init="variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  mask_current=True, mask_current_value=float(-10**5), mask_different_hashes=True, allow_duplicate_attention=False,
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
  :param bool mask_different_hashes: whether a query may only attend to keys with the same hash
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
  if chunks_before is None:
    chunks_before = 1
  if chunks_after is None:
    chunks_after = 0 if past_only else 1
  assert chunks_before >= 0 and chunks_after >= 0
  chunk_stack_offsets_before = list(range(-chunks_before, 0))
  chunk_stack_offsets_after = list(range(1, chunks_after + 1))
  hash_mask_value = 2 ** 31 - 1
  assert hash_mask_value > num_hashes

  # Assume input [B,T|classes?,F|d_model]

  # Create (unnormalized) key/query, value
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
  d[output + '_kq'] = {
    'class': 'copy', 'from': [output + '_qv_split/0']}  # [B,T|classes?,n,F|d_k]
  d[output + '_value'] = {
    'class': 'copy', 'from': [output + '_qv_split/1']}  # [B,T|classes?,n,F|d_v]

  # Mappings from key chunk dim to query chunk dim
  if inside_rec_layer:
    indices_from = ':i'
  else:
    d[output + '_indices'] = {
      'class': 'range_in_axis', 'from': [input], 'axis': time_axis, 'keepdims': False}  # [T|classes]
    indices_from = output + '_indices'
  d[output + '_query_orig_to_sort'] = {
    'class': 'gather', 'from': [output + '_kq_accum_orig_to_sort'], 'position': indices_from,
    'axis': 'stag:rec-history'}  # [B,T|classes?,n] :: T|rec-history
  d[output + '_sort_key_to_query_chunk'] = {
    'class': 'eval', 'from': [output + '_query_orig_to_sort'],
    'eval': key_to_query_chunk, 'out_type': key_to_query_chunk_template,
    'eval_locals': {'chunk_size': chunk_size}}  # [query_chunk_dim?] :: key_chunk_dim
  d[output + '_sort_key_to_query_window'] = {
    'class': 'eval', 'from': [output + '_query_orig_to_sort'],
    'eval': key_to_query_window, 'out_type': key_to_query_window_template,
    'eval_locals': {'chunk_size': chunk_size}}  # [query_window_dim?] :: key_window_dim

  def chunk_accumulated(layer, pad_value, have_feature_dim=False, mode='kq'):
    """
    If mode='kq', d[output + '_kq_accum_' + layer] has shape [X,T|rec-history], will add
     - d[layer + '_key_accum_' + layer + '_chunked'] of shape [X,query_chunk_dim?,2*key_window_dim] and
     - d[layer + '_query_' + layer + '_chunked'] of shape [X,query_chunk_dim?,query_window_dim?].
    If mode='value', d[output + '_value_accum_' + layer] has shape [X,T|rec-history], will add
    d[output + '_value_' + layer + '_chunked'] of shape [X,query_chunk_dim?,2*key_window_dim].

    :param str layer:
    :param float|bool pad_value: for chunking, with which value to pad
    :param bool have_feature_dim: whether the input has a feature dim axis
    :param str mode: whether this is for keys and queries ('kq'), for the values 'value' or for single query ('query').
    """
    assert mode in {'kq', 'value', 'query'}
    _layer = '_' + layer if len(layer) > 0 else ''
    layer_chunked = layer + '_chunked' if len(layer) > 0 else 'chunked'
    key_name = {'kq': 'key', 'value': 'value', 'query': None}[mode]
    # input is [X,T|rec-history]
    if have_feature_dim:
      d[output + '_' + mode + '_accum_' + layer_chunked + '_unnamed'] = {
        'class': 'split_dims', 'from': [output + '_' + mode + '_accum' + _layer], 'pad_value': pad_value,
        'axis': 'stag:rec-history', 'dims': [-1, chunk_size]}  # [X,key_chunk_dim,key_window_dim]
    else:
      d[output + '_' + mode + '_accum_' + layer_chunked + '_feature'] = {
        'class': 'split_dims', 'from': [output + '_' + mode + '_accum' + _layer], 'pad_value': pad_value,
        'axis': 'stag:rec-history', 'dims': [-1, chunk_size]}  # [X,key_chunk_dim,key_window_dim]
      d[output + '_' + mode + '_accum_' + layer_chunked + '_unnamed'] = {
        'class': 'reinterpret_data', 'from': [output + '_' + mode + '_accum_' + layer_chunked + '_feature'],
        'set_axes': {'F': None}}  # [X,key_chunk_dim,key_window_dim]
    if mode == 'kq':
      d[output + '_' + mode + '_accum_' + layer_chunked] = {
        'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
        'from': [output + '_' + mode + '_accum_' + layer_chunked + '_unnamed']}  # [X,key_chunk_dim,key_window_dim]
      d[output + '_' + key_name + '_accum_' + layer_chunked + '_single'] = {
        'class': 'copy',
        'from': [output + '_' + mode + '_accum_' + layer_chunked]}  # [X,key_chunk_dim,key_window_dim]
    elif mode == 'value':
      d[output + '_' + key_name + '_accum_' + layer_chunked + '_single'] = {
        'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
        'from': [output + '_' + mode + '_accum_' + layer_chunked + '_unnamed']}  # [X,key_chunk_dim,key_window_dim]
    elif mode == 'query':
      d[output + '_query_accum_' + layer_chunked] = {
        'class': 'name_axis', 'axis': ['T', 'T+1'], 'description': ['key-chunk', 'key-window'],
        'from': [output + '_' + mode + '_accum_' + layer_chunked + '_unnamed']}  # [X,key_chunk_dim,key_window_dim]
      d[output + '_query_' + layer_chunked + '_all'] = {
        'class': 'gather', 'from': [output + '_query_accum_' + layer_chunked], 'axis': 'stag:key-chunk',
        'position': output + '_sort_key_to_query_chunk'}  # [X,query_chunk_dim?,key_window_dim]
      d[output + '_query_' + layer_chunked] = {
        'class': 'gather', 'from': [output + '_query_' + layer_chunked + '_all'], 'axis': 'stag:key-window',
        'position': output + '_sort_key_to_query_window'}  # [X,query_chunk_dim?,query_window_dim?]
      return  # do not add windows.
    else:
      assert False
    assert key_name is not None
    for stack_offset in chunk_stack_offsets_before + chunk_stack_offsets_after:
      d[output + '_' + key_name + '_accum_' + layer_chunked + '_offset%s' % stack_offset] = {
        'class': 'eval',
        'from': [output + '_' + key_name + '_accum_' + layer_chunked + '_single'],
        'eval': 'tf.roll(source(0), shift=-chunk_offset, axis=source(0, as_data=True).get_axis_from_description("stag:key-chunk"))',  # noqa
        'eval_locals': {'chunk_offset': stack_offset}}  # [X,key_chunk_dim,key_window_dim]
    d[output + '_' + key_name + '_accum_' + layer_chunked + '_stacked_split'] = {
      'class': 'stack',
      'from': (
          [output + '_' + key_name + '_accum_' + layer_chunked + '_offset%s' % i for i in chunk_stack_offsets_before]
          + [output + '_' + key_name + '_accum_' + layer_chunked + '_single']
          + [output + '_' + key_name + '_accum_' + layer_chunked + '_offset%s' % i for i in
            chunk_stack_offsets_after])}  # [X,key_chunk_dim,key_window_dim,chunk_stack_dim]
    d[output + '_' + key_name + '_accum_' + layer_chunked + '_stacked_unnamed'] = {
      'class': 'merge_dims', 'axes': ['stag:key-window', 'spatial:-1'],
      'from': [
        output + '_' + key_name + '_accum_' + layer_chunked + '_stacked_split']}  # [X,key_chunk_dim,2*key_window_dim]
    d[output + '_' + key_name + '_accum_' + layer_chunked + '_stacked'] = {
      'class': 'name_axis', 'axis': ['T+1'], 'description': ['key-stacked-window'],
      'from': [
        output + '_' + key_name + '_accum_' + layer_chunked + '_stacked_unnamed']}  # [X,key_chunk_dim,2*key_window_dim]
    d[output + '_' + key_name + '_accum_' + layer_chunked] = {
      'class': 'gather', 'from': [output + '_' + key_name + '_accum_' + layer_chunked + '_stacked'],
      'axis': 'stag:key-chunk',
      'position': output + '_sort_key_to_query_chunk'}  # [X,query_chunk_dim?,2*key_window_dim]
    if mode == 'kq':
      d[output + '_query_' + layer_chunked + '_all'] = {
        'class': 'gather', 'from': [output + '_' + mode + '_accum_' + layer_chunked], 'axis': 'stag:key-chunk',
        'position': output + '_sort_key_to_query_chunk'}  # [X,query_chunk_dim?,key_window_dim]
      d[output + '_query_' + layer_chunked] = {
        'class': 'gather', 'from': [output + '_query_' + layer_chunked + '_all'], 'axis': 'stag:key-window',
        'position': output + '_sort_key_to_query_window'}  # [X,query_chunk_dim?,query_window_dim?]

  # Hash the key/query
  make_lsh_hash_gen(
    d, output + '_hash_gen', key_dim=key_dim, num_hashes=num_hashes, num_heads=num_heads, num_rounds=num_rounds,
    hash_init=ff_init)  # [B,n,r,d_k,F|d_h]
  apply_lsh_hash_gen(
    d, input=output + '_kq', hash_gen_input=output + '_hash_gen', output=output + '_kq_hash',
    time_axis=time_axis, num_hashes=num_hashes, hash_mask_value=None)  # [B,T|classes?,n,r] :: d_h

  # Accumulate all past hashes
  d[output + '_kq_accum_hash_unsorted_unmasked'] = {
    'class': 'cum_concat', 'from': [output + '_kq_hash']}  # [B,T|rec-history,n,r] :: d_h
  d[output + '_kq_accum_hash_unsorted'] = {
    'class': 'seq_len_mask', 'from': [output + '_kq_accum_hash_unsorted_unmasked'],
    'axis': 'stag:rec-history',
    'mask_value': hash_mask_value}  # [B,T|rec-history,n,r] :: d_h

  # Compute a permutation by looking at the unsorted hashes
  d[output + '_kq_accum_sort_to_orig'] = {
    'class': 'eval', 'eval': argsort_eval % 'stag:rec-history',
    'from': [output + '_kq_accum_hash_unsorted']}  # [B,T|rec-history,n,r] :: T|rec-history
  chunk_accumulated('sort_to_orig', pad_value=hash_mask_value)

  # Sort the hashes themselves
  d[output + '_kq_accum_hash'] = {
    'class': 'gather', 'from': [output + '_kq_accum_hash_unsorted'],
    'axis': 'stag:rec-history',
    'position': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,r] :: d_h
  chunk_accumulated('hash', pad_value=hash_mask_value)

  # Invert permutation to undo sorting later
  d[output + '_kq_accum_orig_indices'] = {
    'class': 'range_in_axis', 'from': [output + '_kq_accum_hash_unsorted'], 'axis': 'stag:rec-history',
    'keepdims': False}  # [T|rec-history] :: T|rec-history
  d[output + '_kq_accum_orig_to_sort'] = {
    'class': 'scatter_nd', 'from': [output + '_kq_accum_orig_indices'],
    'position': output + '_kq_accum_sort_to_orig', 'position_axis': 'stag:rec-history',
    'output_dim_via_time_from': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,r] :: T|rec-history
  d[output + '_kq_accum_orig_chunk_to_sort'] = {
    'class': 'eval', 'eval': 'tf.math.floordiv(source(0), %s)' % chunk_size,
    'from': [output + '_kq_accum_orig_to_sort']}  # [B,T|rec-history,n,r] :: key/query_chunk_dim

  # Accumulate all past keys/queries
  d[output + '_kq_accum_unsorted'] = {
    'class': 'cum_concat', 'from': [output + '_kq']}  # [B,T|rec-history,n,F|d_k]
  d[output + '_kq_accum'] = {
    'class': 'gather', 'from': [output + '_kq_accum_unsorted'],
    'axis': 'stag:rec-history',
    'position': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,r,F|d_k]

  # Compute attention keys and queries
  chunk_accumulated('', pad_value=0, have_feature_dim=True, mode='kq')
  # Override that attention keys are set to normalized attention queries
  d[output + '_key_accum_chunked_single'] = {
    'class': 'eval',
    'eval': normalize_eval,
    'from': [output + '_kq_accum_chunked']}  # [key_chunk_dim,key_window_dim,B,n,r,F|d_k]
  d[output + '_query_chunked_scaled'] = {
    'class': 'eval', 'eval': '%s * source(0)' % (key_dim ** -0.5),
    'from': [output + '_query_chunked']}  # [B,query_chunk_dim?,query_window_dim?,F|d_k]

  # Compute energy mask
  masking_layers_from = []
  masking_layers_have_query_window = False
  if past_only:
    d[output + '_energy_chunked_mask_past_only'] = {
      'class': 'compare',
      'from': [output + '_query_sort_to_orig_chunked', output + '_key_accum_sort_to_orig_chunked'],
      'kind': 'greater_equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
    masking_layers_from.append(output + '_energy_chunked_mask_past_only')
    masking_layers_have_query_window = True
  if mask_different_hashes:
    # Masking valid positions is not necessary in this case as invalid positions will be masked with a special
    # hash value
    d[output + '_energy_chunked_mask_matching_hash'] = {
      'class': 'compare',
      'from': [output + '_query_hash_chunked', output + '_key_accum_hash_chunked'],
      'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
    masking_layers_from.append(output + '_energy_chunked_mask_matching_hash')
    masking_layers_have_query_window = True
  else:
    d[output + '_energy_chunked_mask_valid_key_position'] = {
      'class': 'compare',
      'from': [output + '_key_accum_hash_chunked'], 'value': hash_mask_value,
      'kind': 'not_equal'}  # [B,query_chunk_dim?,2*key_window_dim,n,r]
    masking_layers_from.append(output + '_energy_chunked_mask_valid_key_position')
  if len(masking_layers_from) > 1:
    d[output + '_energy_chunked_mask'] = {
      'class': 'compare',
      'from': masking_layers_from,
      'kind': 'logical_and'}  # [B,query_chunk_dim?,[query_window_dim?]?,2*key_window_dim,n,r]
  else:
    d[output + '_energy_chunked_mask'] = {
      'class': 'copy', 'from': masking_layers_from}  # [B,query_chunk_dim?,[query_window_dim?]?,2*key_window_dim,n,r]

  # Compute number of duplicates of each key
  if not allow_duplicate_attention:
    d[output + '_key_accum_sort_to_orig_sorted'] = {
      'class': 'eval',
      'from': [output + '_key_accum_sort_to_orig_chunked', output + '_kq_accum_orig_chunk_to_sort', output + '_value'],
      'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,n,T|classes?,r,2*key_window_dim] :: T|rec-history
    d[output + '_key_accum_sort_to_orig_sorted_other'] = {
      'class': 'name_axis', 'axis': ['stag:att-round', 'stag:key-stacked-window'],
      'description': ['stag:att-round-other', 'stag:key-stacked-window-other'],
      'from': [output + '_key_accum_sort_to_orig_sorted']}  # [B,n,T|classes?,r',2*key_window_dim'] :: T|rec-history

    if masking_layers_have_query_window:
      d[output + '_energy_chunked_mask_flattened'] = {
        'class': 'merge_dims', 'axes': ['stag:query-chunk?', 'stag:query-window?'],
        'from': [output + '_energy_chunked_mask']}  # [B,query_chunk_dim?*query_window_dim?,n,r,2*key_window_dim]
      d[output + '_energy_chunked_mask_sorted'] = {
        'class': 'eval',
        'from': [output + '_energy_chunked_mask_flattened', output + '_kq_accum_orig_to_sort', output + '_value'],
        'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,n,T|classes?,n,r,2*key_window_dim]
    else:
      d[output + '_energy_chunked_mask_sorted'] = {
        'class': 'eval',
        'from': [output + '_energy_chunked_mask', output + '_kq_accum_orig_chunk_to_sort', output + '_value'],
        'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,n,T|classes?,n,r,2*key_window_dim]
    d[output + '_energy_chunked_mask_sorted_other'] = {
      'class': 'name_axis', 'axis': ['stag:att-round', 'stag:key-stacked-window'],
      'description': ['stag:att-round-other', 'stag:key-stacked-window-other'],
      'from': [output + '_energy_chunked_mask_sorted']}  # [B,n,T|classes?,r',2*key_window_dim']
    d[output + '_query_duplicates_sorted_compare_unmasked_time'] = {
      'class': 'compare',
      'from': [output + '_key_accum_sort_to_orig_sorted', output + '_key_accum_sort_to_orig_sorted_other'],
      'kind': 'equal'}  # [B,n,T|classes?,r,2*key_window_dim,r',2*key_window_dim']
    d[output + '_query_duplicates_sorted_compare_unmasked'] = {
      'class': 'reinterpret_data', 'from': [output + '_query_duplicates_sorted_compare_unmasked_time'],
      'set_axes': {'T': None}}  # [B,n,T|classes?,r,2*key_window_dim,r',2*key_window_dim']
    d[output + '_query_duplicates_sorted_compare'] = {
      'class': 'compare',
      'from': [
        output + '_query_duplicates_sorted_compare_unmasked', output + '_energy_chunked_mask_sorted_other'],
      'kind': 'logical_and'}  # [B,n,T|classes?,r,2*key_window_dim,r',2*key_window_dim']
    d[output + '_query_duplicates_sorted_compare_float'] = {
      'class': 'cast', 'from': [output + '_query_duplicates_sorted_compare'],
      'dtype': 'float32'}  # [B,n,T|classes?,r,2*key_window_dim,r',2*key_window_dim']
    d[output + '_query_duplicates_sorted'] = {
      'class': 'reduce', 'mode': 'sum', 'from': [output + '_query_duplicates_sorted_compare_float'],
      'axes': ['stag:att-round-other', 'stag:key-stacked-window-other']}  # [B,n,r,T|classes?,2*key_window_dim]
    d[output + '_query_accum_duplicates_sorted'] = {
      'class': 'cum_concat', 'axis': time_axis,
      'from': [output + '_query_duplicates_sorted']}  # [B,n,r,T|rec-history,2*key_window_dim]
    d[output + '_query_accum_duplicates'] = {
      'class': 'gather', 'from': [output + '_query_accum_duplicates_sorted'], 'axis': 'stag:rec-history',
      'position': output + '_kq_accum_sort_to_orig'}  # [B,n,r,T|rec-history,2*key_window_dim]
    chunk_accumulated('duplicates', pad_value=0.0, mode='query')

  # Compute energy small mask (i.e. the entries that will be set to mask_current_value)
  small_masking_layers_from = [output + '_energy_chunked_mask_small_invalid_query_position']
  # We never want the attention weights to be NaN for any query (even for unmasked queries),
  # and thus need to have at least one masked key for every query.
  # Otherwise, the gradients will be NaN.
  # We ensure this by masking all energies with a small (finite) number.
  d[output + '_energy_chunked_mask_small_invalid_query_position'] = {
    'class': 'compare',
    'from': [output + '_query_hash_chunked'], 'value': hash_mask_value,
    'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,n,r]
  if mask_current:
    d[output + '_energy_chunked_mask_small_current'] = {
      'class': 'compare',
      'from': [output + '_query_sort_to_orig_chunked', output + '_key_accum_sort_to_orig_chunked'],
      'kind': 'equal'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
    small_masking_layers_from.append(output + '_energy_chunked_mask_small_current')
  if len(small_masking_layers_from) > 1:
    d[output + '_energy_chunked_mask_small'] = {
      'class': 'compare',
      'from': small_masking_layers_from,
      'kind': 'logical_or'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  else:
    d[output + '_energy_chunked_mask_small'] = {
      'class': 'copy', 'from': small_masking_layers_from}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]

  # Compute energy
  d[output + '_energy_chunked_feature'] = {
    'class': 'dot', 'red1': 'static:-1', 'red2': 'static:-1',
    'var1': 'stag:query-window?', 'var2': 'stag:key-stacked-window',
    'from': [output + '_query_chunked_scaled', output + '_key_accum_chunked'],
    'debug': True}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  d[output + '_energy_chunked_unmasked_duplicates'] = {
    'class': 'reinterpret_data', 'from': [output + '_energy_chunked_feature'],
    'set_axes': {'F': None}}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  if allow_duplicate_attention:
    d[output + '_energy_chunked_unmasked'] = {
      'class': 'copy',
      'from': [output + '_energy_chunked_unmasked_duplicates']}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]  # noqa
  else:
    d[output + '_energy_chunked_unmasked'] = {
      'class': 'eval',
      'from': [output + '_energy_chunked_unmasked_duplicates', output + '_query_duplicates_chunked'],
      'eval': 'source(0) - tf.math.log(source(1) + 1e-9)'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]  # noqa
  d[output + '_energy_chunked_not_small'] = {  # broken!! has weird time dim
    'class': 'switch', 'condition': output + '_energy_chunked_mask',
    'true_from': output + '_energy_chunked_unmasked',
    'false_from': float('-inf')}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  d[output + '_energy_chunked'] = {
    'class': 'switch', 'condition': output + '_energy_chunked_mask_small',
    'true_from': mask_current_value,
    'false_from': output + '_energy_chunked_not_small'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  d[output + '_energy_chunked_logsumexp'] = {
    'class': 'reduce', 'mode': 'logsumexp', 'axis': 'stag:key-stacked-window',
    'from': [output + '_energy_chunked']}  # [B,query_chunk_dim?,query_window_dim?,n,r]
  d[output + '_weights_chunked'] = {
    'class': 'eval', 'from': [output + '_energy_chunked', output + '_energy_chunked_logsumexp'],
    'eval': 'tf.exp(source(0) - source(1))'}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]
  d[output + '_weights_chunked_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights_chunked'],
    'dropout': dropout}  # [B,query_chunk_dim?,query_window_dim?,2*key_window_dim,n,r]

  # Compute attention values
  d[output + '_value_accum_unsorted'] = {
    'class': 'cum_concat', 'from': [output + '_value']}  # [B,T|rec-history,n,r,F|d_v]
  d[output + '_value_accum'] = {
    'class': 'gather', 'from': [output + '_value_accum_unsorted'],
    'axis': 'stag:rec-history',
    'position': output + '_kq_accum_sort_to_orig'}  # [B,T|rec-history,n,r,F|d_v]
  chunk_accumulated('', pad_value=0, have_feature_dim=True, mode='value')

  # Compute the outputted weighted sum (i.e. the context vector)
  d[output + '_round_output_chunked'] = {
    'class': 'dot', 'red1': 'stag:key-stacked-window', 'red2': 'stag:key-stacked-window',
    'var1': 'stag:query-window?', 'var2': 'static:-1', 'debug': True,
    'from': [
      output + '_weights_chunked_drop',
      output + '_value_accum_chunked']}  # [B,query_chunk_dim?,query_window_dim?,n,r,F|d_v]
  d[output + '_round_output_sorted'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk?', 'stag:query-window?'],
    'from': [output + '_round_output_chunked']}  # [B,query_chunk_dim?*query_window_dim?,n,r,F|d_v]
  d[output + '_round_output'] = {
    'class': 'eval', 'from': [output + '_round_output_sorted', output + '_kq_accum_orig_to_sort', output + '_value'],
    'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,T|classes?,n,r,F|d_v]
  d[output + '_energy_logsumexp_sorted'] = {
    'class': 'merge_dims', 'axes': ['stag:query-chunk?', 'stag:query-window?'],
    'from': [output + '_energy_chunked_logsumexp']}  # [B,query_chunk_dim?*query_window_dim?,n,r]
  d[output + '_energy_logsumexp'] = {
    'class': 'eval',
    'from': [output + '_energy_logsumexp_sorted', output + '_kq_accum_orig_to_sort', output + '_value'],
    'eval': maybe_gather, 'out_type': maybe_gather_template}  # [B,T|classes?,n,r]
  d[output + '_round_output_weights'] = {
    'class': 'softmax_over_spatial', 'axis': 'stag:att-rounds', 'use_time_mask': False, 'energy_factor': 1.0,
    'from': [output + '_energy_logsumexp']}  # [B,T|classes?,n,r]
  d[output + '_round_output_weighted'] = {
    'class': 'combine', 'kind': 'mul',
    'from': [output + '_round_output_weights', output + '_round_output']}  # [B,T|classes?,n,F|d_v]
  d[output + '_output'] = {
    'class': 'reduce', 'axis': 'stag:att-rounds', 'mode': 'sum',
    'from': [output + '_round_output_weighted']}  # [B,T|classes?,n,F|d_v]
  d[output + '_output_unnamed'] = {
    'class': 'name_axis', 'axis': 'stag:att-heads', 'description': None,
    'from': [output + '_output']}  # [B,T|classes?,F|n*d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static',
    'from': [output + '_output_unnamed']}  # [B,T|classes?,F|n*d_v]

  if debug_print:
    for name in [output + n for n in [
      '_kq', '_value', '_sort_key_to_query_chunk', '_sort_key_to_query_window',
      '_kq_hash', '_kq_accum_hash_unsorted', '_kq_accum_hash', '_kq_accum_hash_chunked',
      '_key_accum_hash_chunked', '_query_hash_chunked', '_kq_accum_sort_to_orig', '_kq_accum_orig_to_sort',
      '_kq_accum_unsorted', '_kq_accum', '_kq_accum_chunked', '_key_accum_chunked', '_query_chunked',
      '_energy_chunked_unmasked', '_energy_chunked_unmasked_duplicates', '_energy_chunked_not_small',
      '_query_sort_to_orig_chunked', '_key_accum_sort_to_orig_chunked',
      '_energy_chunked_mask', '_energy_chunked_mask_small',
      '_energy_chunked', '_weights_chunked', '_value_accum_unsorted', '_value_accum', '_value_accum_chunked',
      '_round_output_chunked', '_round_output_sorted', '_output', '_att'
    ] + (['_query_duplicates_chunked', '_energy_chunked_mask_sorted', '_key_accum_sort_to_orig_sorted',
      '_key_accum_sort_to_orig_sorted_other', '_query_duplicates_sorted', '_query_duplicates_sorted_compare_unmasked'] if not allow_duplicate_attention else [])] \
      + masking_layers_from + small_masking_layers_from:
      d[name + '_orig'] = d[name]
      d[name] = {'class': 'print', 'from': [name + '_orig']}
      #d[name + '_print'] = {
      #  'class': 'print', 'from': [name], 'is_output_layer': True}


def dump_lsh_self_attention_weights(d, output, file_name):
  """
  Uses a HDF dump layer to extract LSH attention weights.

  :param dict[str,dict] d: the network dict to write into
  :param str output: prefix name of the lsh layers
  :param str file_name: HDF file
  """
  d[output + '_weights_chunked_dump_length'] = {
    'class': 'length', 'from': [output + '_kq']}  # [B,T|classes]
  d[output + '_weights_chunked_dump'] = {
    'class': 'hdf_dump', 'from': [output + '_weights_chunked'],
    'extra': {
      'orig_to_sort': output + '_kq_accum_orig_to_sort',
      'hash': output + '_kq_hash',
      'seq_length': output + '_weights_chunked_dump_length'},
    'filename': file_name,
    'is_output_layer': True}


def add_vanilla_self_attention_layer(
  d, input, output, inside_rec_layer=True, past_only=None, time_axis=None,
  num_heads=8, key_dim=64, value_dim=64, dropout=0.0,
  ff_init = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=%s)" % 1.0,
  share_key_query=False, normalize_keys=None,
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
  if normalize_keys is None:
    normalize_keys = share_key_query

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
    if normalize_keys:
      d[output + '_key'] = {
        'class': 'eval', 'eval': normalize_eval, 'from': [output + '_qkv_split/1']}  # [B,T?,n,F|d_k]
    else:
      d[output + '_key'] = {
        'class': 'copy', 'from': [output + '_qkv_split/1']}  # [B,T?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qkv_split/2']}  # [B,T?,n,F|d_v]
  else:  # share_key_query
    d[output + '_qv0'] = {
      'class': 'linear', 'activation': None, 'with_bias': False, 'from': [input],
      'n_out': num_heads * (key_dim + value_dim), 'forward_weights_init': ff_init}  # [B,T?,F|n*(d_k+d_v)]
    d[output + '_qv'] = {
      'class': 'split_dims', 'axis': 'F', 'dims': (num_heads, key_dim + value_dim),
      'from': [output + '_qv0']}  # [B,T?,n,F|d_k+d_v]
    d[output + '_qv_split'] = {
      'class': 'split', 'axis': 'F', 'size_splits': (key_dim, value_dim),
      'from': [output + '_qv']}
    d[output + '_query'] = {
      'class': 'copy', 'from': [output + '_qv_split/0']}  # [B,T?,n,F|d_k]
    if normalize_keys:
      d[output + '_key'] = {
        'class': 'eval', 'eval': normalize_eval, 'from': [output + '_query']}  # [B,T?,n,F|d_k]
    else:
      d[output + '_key'] = {'class': 'copy', 'from': [output + '_query']}  # [B,T?,n,F|d_k]
    d[output + '_value'] = {
      'class': 'copy', 'from': [output + '_qv_split/1']}  # [B,T?,n,F|d_v]

  # Accumulate keys/values or rename the axis
  if inside_rec_layer:
    d[output + '_key_accum'] = {
      'class': 'cum_concat', 'from': [output + '_key']}  # [B,T|rec-history,n,F|d_k]
    d[output + '_value_accum'] = {
      'class': 'cum_concat', 'from': [output + '_value']}  # [B,T|rec-history,n,F|d_v]
    key_axis = 'stag:rec-history'
  else:
    key_dim_tag = DimensionTag(kind=DimensionTag.Types.Time, description='self-att-keys')
    d[output + '_key_accum'] = {
      'class': 'reinterpret_data', 'set_dim_tags': {time_axis: key_dim_tag},
      'from': [output + '_key']}  # [B,T|keys,n,F|d_k]
    d[output + '_value_accum'] = {
      'class': 'reinterpret_data', 'set_dim_tags': {time_axis: key_dim_tag},
      'from': [output + '_value']}  # [B,T|keys,n,F|d_v]
    key_axis = 'stag:' + key_dim_tag.description

  # Calculate the energies
  d[output + '_energy'] = {
    'class': 'dot', 'from': [output + '_query', output + '_key_accum'],
    'red1': 'static:-1', 'red2': 'static:-1', 'var1': time_axis + '?', 'var2': key_axis}  # [B,n,T?,T|rec-history]

  need_indices = past_only or mask_current
  if need_indices:
    if inside_rec_layer:
      query_indices_from = ':i'
    else:
      d[output + '_query_indices'] = {'class': 'range_in_axis', 'from': [input], 'axis': time_axis,
        'keepdims': False}  # [T]
      query_indices_from = output + '_query_indices'
    d[output + '_key_accum_indices'] = {
      'class': 'range_in_axis', 'from': [output + '_key_accum'], 'axis': key_axis,
      'keepdims': False}  # [T|rec-history]
  if past_only:
    d[output + '_energy_unmasked'] = d[output + '_energy']
    d[output + '_energy_mask'] = {
      'class': 'compare', 'kind': 'greater_equal', 'from': [query_indices_from, output + '_key_accum_indices']}
    d[output + '_energy'] = {
      'class': 'switch', 'true_from': output + '_energy_unmasked', 'false_from': float('-inf'),
      'condition': output + '_energy_mask'}  # [B,n,T?,T|rec-history]
  if mask_current:
    d[output + '_energy_unmasked_current'] = d[output + '_energy']
    d[output + '_energy_mask_current'] = {
      'class': 'compare', 'kind': 'equal', 'from': [query_indices_from, output + '_key_accum_indices']}
    d[output + '_energy'] = {
      'class': 'switch', 'true_from': mask_current_value, 'false_from': output + '_energy_unmasked_current',
      'condition': output + '_energy_mask_current'}  # [B,n,T?,T|rec-history]

  # If past_only=True, do not apply a time mask here, as we apply our own masking using energy_mask.
  # If we would apply additional masking here, we would mask away all keys for queries that are unmasked, giving
  # attention weights NaN for these queries. Even though these are masked away later in the forward pass, the gradient
  # can still become NaN.
  # If past_only=False, do apply the normal time mask.
  d[output + '_weights'] = {
    'class': 'softmax_over_spatial', 'from': [output + '_energy'], 'axis': key_axis,
    'energy_factor': key_dim ** -0.5,
    'use_time_mask': not past_only}  # [B,n,T?,T|rec-history]
  d[output + '_weights_drop'] = {
    'class': 'dropout', 'dropout_noise_shape': {'*': None}, 'from': [output + '_weights'],
    'dropout': dropout}  # [B,n,T?,T|rec-history]

  d[output + '_output'] = {
    'class': 'dot', 'from': [output + '_weights_drop', output + '_value_accum'],
    'red1': key_axis, 'red2': key_axis, 'var1': time_axis + '?', 'var2': 'static:-1'}  # [B,n,T?,F|d_v]
  d[output + '_att'] = {
    'class': 'merge_dims', 'axes': 'static', 'from': [output + '_output']}  # [B,T?,F|n*d_v]
