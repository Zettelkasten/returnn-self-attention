import re

import tensorflow as tf
from typing import List, Union
from returnn.extern import graph_editor as ge

# save original gradients
from tensorflow.python.ops import gradients as tf_gradients_lib
tf_gradients = tf_gradients_lib.gradients


def recomputed_tf_gradients(recompute_ops_regex: str,
                            ys: Union[List[tf.Tensor], tf.Tensor], xs: Union[List[tf.Tensor], tf.Tensor], grad_ys=None,
                            **kwargs) -> List[tf.Tensor]:
    if not isinstance(ys, list):
      ys: List[tf.Tensor] = [ys]
    if not isinstance(xs, list):
      xs: List[tf.Tensor] = [xs]
    graph = ys[0].graph

    # compute gradients normally
    before_ops = graph.get_operations()
    x_grads = tf_gradients(ys=ys, xs=xs, grad_ys=grad_ys, **kwargs)
    all_grad_ops: List[tf.Operation] = list(set(graph.get_operations()) - set(before_ops))
    assert all(x_grad.op in all_grad_ops for x_grad in x_grads)

    bwd_ops_from_x_grads: List[tf.Operation] = ge.get_backward_walk_ops([y.op for y in ys], inclusive=True)
    fwd_ops: List[tf.Operation] = ge.get_forward_walk_ops(
        [x.op for x in xs], within_ops=bwd_ops_from_x_grads, inclusive=True)

    sanity_checks = False

    recompute_ops = [op for op in fwd_ops if re.match(recompute_ops_regex, op.name) and len(op.inputs) > 0]
    assert len(recompute_ops) > 0, 'bad regex %r' % recompute_ops_regex

    # make a copy of recompute fwd pass tensors
    with tf.name_scope('recompute_fwd_pass'):
        if sanity_checks:
            assert all(recompute_op not in recompute_ops for recompute_op in recompute_ops)
            assert all(x.op in fwd_ops and x.op not in recompute_ops for x in xs)
            assert all(y.op in fwd_ops and y.op not in recompute_ops for y in ys)
            assert all(len(op.outputs) >= 1 for op in recompute_ops)
        recompute_ts: List[tf.Tensor] = [t for op in recompute_ops for t in op.outputs]
        print('Recomputing forward pass tensors matching %r in backward pass: %r' % (recompute_ops_regex, recompute_ts))
        _, info = ge.copy_with_input_replacements(ge.sgv(recompute_ops), {})
        copied_recompute_ts: List[tf.Tensor] = [info._transformed_ts[recompute_t] for recompute_t in recompute_ts]
        copied_recompute_ops = [compied_recompute_t.op for compied_recompute_t in copied_recompute_ts]
        control_ops = [y.op for y in ys]
        for copied_recompute_op in copied_recompute_ops:
            ge.add_control_inputs(
                op=copied_recompute_op,
                cops=[control_op for control_op in control_ops if control_op not in copied_recompute_op.control_inputs])

    # replace gradients by swapping in recomputed fwd passes
    assert len(recompute_ts) == len(copied_recompute_ts)
    num_replacements = ge.reroute_ts(copied_recompute_ts, recompute_ts, can_modify=all_grad_ops)
    assert num_replacements > 0
    print('Recomputing %r operations in total' % num_replacements)

    # Some sanity checks
    if sanity_checks:
        bwd_ops_from_x_grads: List[tf.Operation] = ge.get_backward_walk_ops(
            [x_grad.op for x_grad in x_grads],
            within_ops=all_grad_ops + recompute_ops + copied_recompute_ops, inclusive=True)
        # loss does not directly depend on recompute_ops
        assert all(recompute_op not in bwd_ops_from_x_grads for recompute_op in recompute_ops)
        # loss does however depend on copied_recompute_ops for at least some ops
        assert any(copied_recompute_op in bwd_ops_from_x_grads for copied_recompute_op in copied_recompute_ops)
        # copied_recompute_op does not depend on recompute_op
        assert all(
            t not in ge.get_backward_walk_ops([copied_t]) for t, copied_t in zip(recompute_ts, copied_recompute_ts))
        # no copied_recompute_op directly depends on a recompute_op
        for recompute_op, copied_recompute_op in zip(recompute_ops, copied_recompute_ops):
            copied_op_deps = ge.get_backward_walk_ops(
                [copied_recompute_op], within_ops=all_grad_ops + copied_recompute_ops + recompute_ops)  # all but non recomputed fwd ops
            assert all(other_recompute_op not in copied_op_deps for other_recompute_op in recompute_ops)

    return x_grads


def register_recomputed_gradients(recompute_ops_regex: str):
    def _tf_gradients(ys: Union[List[tf.Tensor], tf.Tensor], xs: Union[List[tf.Tensor], tf.Tensor], grad_ys=None,
                      **kwargs) -> List[tf.Tensor]:
        return recomputed_tf_gradients(recompute_ops_regex=recompute_ops_regex, ys=ys, xs=xs, grad_ys=grad_ys, **kwargs)

    from tensorflow.python.ops import gradients as tf_gradients
    tf_gradients.gradients = _tf_gradients
