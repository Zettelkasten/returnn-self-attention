import sys
sys.path.insert(1, '/u/petrick/software/returnn/official/')
sys.path.insert(1, '/u/petrick/software/returnn/official/tests')
sys.path.insert(1, '/u/petrick/lsh/playground/encoder_lsh')

from returnn.util import better_exchook
import unittest
from pprint import pprint
from nose.tools import assert_equal, assert_is_instance
from tests.test_TFNetworkLayer import make_scope, make_feed_dict
from tests.test_TFNetworkRecLayer import check_reclayer_optimize_out
from returnn.config import Config
from returnn.tf.network import *
from returnn.tf.layers.basic import *

from lsh_attention import *


# TODO: Currently fails. Why?
def test_lsh_attention_optimize_out():
  network = {}
  add_lsh_self_attention_layer(
    network, 'data:source', 'att', chunks_before=1, chunks_after=0, chunk_size=10, debug_print=True,
    num_heads=2, key_dim=3, value_dim=3, num_hashes=26, inside_rec_layer=True, past_only=True, time_axis='stag:extern_data:data')

  check_reclayer_optimize_out(
    {'class': 'linear', 'from': 'att_att', 'activation': None},
    other_subnet_layers=network)


def _test_lsh_self_attention_no_mask_different_hashes(
    n_time, past_only, mask_current, chunk_size, chunks_before, chunks_after, n_batch=3):
  print(
    'Testing n_time =', n_time, 'past_only =', past_only, 'mask_current =', mask_current, 'chunk_size =', chunk_size,
    'chunks_before =', chunks_before, 'chunks_after =', chunks_after)
  with make_scope() as session:
    total_chunks = chunks_before + 1 + chunks_after
    assert n_time <= chunk_size * (chunks_before + 1 + chunks_after), (
      'if chunk size is too small, vanilla attention != lsh attention')
    assert n_time >= chunk_size * (total_chunks - 1), 'if chunk size is too big, we might attend multiple times'
    num_heads, key_dim, value_dim = 2, 3, 3
    net_dict = {
      "lsh_out": {"class": "copy", "from": "lsh_att", "is_output_layer": True},  # [B,T1,F]
      "vanilla_out": {"class": "copy", "from": "vanilla_att", "is_output_layer": True},  # [B,T1,F]

      "output": {"class": "copy", "from": ["lsh_att"]}}  # [B,T,F]
    add_lsh_self_attention_layer(
      net_dict, 'data', 'lsh', inside_rec_layer=False, past_only=past_only,
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, num_hashes=26,
      chunk_size=chunk_size, chunks_before=chunks_before, chunks_after=chunks_after,
      mask_current=mask_current, mask_different_hashes=False, debug_print=True)
    add_vanilla_self_attention_layer(
      net_dict, 'data', 'vanilla', inside_rec_layer=False, past_only=past_only,
      num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, share_key_query=True,
      mask_current=mask_current
    )
    net_dict["vanilla_qv0"]["reuse_params"] = "lsh_qv0"

    config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
    config.update(dict(num_inputs=num_heads*key_dim, num_outputs=num_heads*value_dim))
    network = TFNetwork(config=config, train_flag=True)
    network.construct_from_dict(net_dict)

    assert_equal(network.get_layer("vanilla_out").output.shape, (None, num_heads * value_dim))
    assert_equal(network.get_layer("lsh_out").output.shape, (None, num_heads * value_dim))
    feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True, n_batch=n_batch, n_time=n_time)
    session.run(tf_compat.v1.global_variables_initializer())

    input_out = network.get_layer("data").output
    vanilla_out = network.get_layer("vanilla_out").output
    lsh_out = network.get_layer("lsh_out").output
    vanilla, lsh, sizes, vanilla_sizes, lsh_sizes = session.run(
      [vanilla_out.placeholder, lsh_out.placeholder,
        input_out.size_placeholder[input_out.time_dim_axis_excluding_batch],
        vanilla_out.size_placeholder[vanilla_out.time_dim_axis_excluding_batch],
        lsh_out.size_placeholder[lsh_out.time_dim_axis_excluding_batch]],
      feed_dict=feed_dict)
    numpy.testing.assert_equal(vanilla_sizes, sizes)
    numpy.testing.assert_equal(lsh_sizes, sizes)
    # take into account different seq lengths
    assert vanilla_out.batch_dim_axis == lsh_out.batch_dim_axis == 0
    assert vanilla_out.time_dim_axis == lsh_out.time_dim_axis == 1
    mask = (numpy.arange(numpy.shape(vanilla)[1]).reshape([1,-1,1]) < sizes.reshape([-1,1,1]))
    vanilla = vanilla * mask
    lsh = lsh * mask
    print('seq lengths:', sizes)
    print('vanilla:  - ', vanilla_out)
    pprint(vanilla)
    print('lsh:  -', lsh_out)
    pprint(lsh)
    numpy.testing.assert_almost_equal(vanilla, lsh, decimal=5)
    print('They are equal!')


def test_lsh_self_attention_no_mask_different_hashes():
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=False, chunk_size=5, chunks_before=1, chunks_after=1)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=False, mask_current=True, chunk_size=5, chunks_before=1, chunks_after=1)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=False, chunk_size=7, chunks_before=1, chunks_after=0)
  _test_lsh_self_attention_no_mask_different_hashes(
    n_time=13, past_only=True, mask_current=True, chunk_size=7, chunks_before=1, chunks_after=0)


def test_vanilla_self_attention_equal_to_SelfAttentionLayer():
  for past_only in [False, True]:
    with make_scope() as session:
      print('Testing past_only=%s' % past_only)
      n_time = 13
      num_heads, key_dim, value_dim = 2, 3, 3
      net_dict = {
        "single_layer_att": {
          "class": "self_attention", "from": "data", "num_heads": num_heads, "total_key_dim": num_heads * key_dim,
          "n_out": num_heads * value_dim, "attention_left_only": past_only, 'is_output_layer': True},  # [B,T,F]
        "multi_layer_att": None  # [B,T,F], added below.
      }
      add_vanilla_self_attention_layer(
        net_dict, 'data', 'multi_layer', inside_rec_layer=False, past_only=past_only,
        num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
      net_dict["multi_layer_att"]["is_output_layer"] = True
      def custom(reuse_layer, *args, **kwargs):
        return tf.identity(reuse_layer.params['QKV'])
      net_dict["multi_layer_qkv0"]["reuse_params"] = {
        "auto_create_missing": False, "map": {"W": {"reuse_layer": "single_layer_att", "custom": custom}}}

      config = Config({"debug_print_layer_output_template": True, "debug_add_check_numerics_ops": True})
      config.update(dict(num_inputs=num_heads*key_dim, num_outputs=num_heads*value_dim))
      network = TFNetwork(config=config, train_flag=True)
      network.construct_from_dict(net_dict)

      single_out = network.get_layer("single_layer_att").output
      multi_out = network.get_layer("multi_layer_att").output
      assert_equal(single_out.shape, (None, num_heads * value_dim))
      assert_equal(multi_out.shape, (None, num_heads * value_dim))
      feed_dict = make_feed_dict(network.extern_data.data.values(), same_time=True, n_time=n_time)
      session.run(tf_compat.v1.global_variables_initializer())

      single, multi = session.run([single_out.placeholder, multi_out.placeholder], feed_dict=feed_dict)
      print('single layer output:')
      pprint(single)
      print('multi layer output:')
      pprint(multi)
      numpy.testing.assert_almost_equal(single, multi, decimal=5)
      print('They are equal!')

if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    import threading
    #if len(list(threading.enumerate())) > 1:
    #  print("Warning, more than one thread at exit:")
    #  better_exchook.dump_all_thread_tracebacks()

