"""Post process for musa quantized model"""
# pylint: disable=unused-variable
import torch
from torch import fx


def _getattr(model, name):
    """customize getattr function to recursive get attribute for pytorch module"""
    name_list = name.split(".")
    for name in name_list:  # pylint: disable=redefined-argument-from-local
        model = getattr(model, name)
    return model


def postprocess(model: fx.GraphModule):
    """replace float add to quantized one, reduce dequantize ops"""
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        if not isinstance(_getattr(model, node.target), torch.nn.Upsample):
            continue
        args = node.args
        assert len(args) == 1
        up_arg = args[0]
        if up_arg.target == "dequantize":
            quant_arg = up_arg.args[0]
        else:
            continue
        assert len(node.users) == 1
        cat_node = list(node.users.keys())[0]
        assert cat_node.target == torch.cat
        cat_inputs = cat_node.args[0]
        flag = True
        for cat_inp in cat_inputs:
            if cat_inp is node:
                continue
            if cat_inp.target == "dequantize":
                cat_quant = cat_inp.args[0]
                cat_inp.replace_all_uses_with(cat_quant)
                model.graph.erase_node(cat_inp)
            else:
                flag = False
        if flag and len(cat_node.users) == 1:
            up_arg.replace_all_uses_with(quant_arg)
            model.graph.erase_node(up_arg)
            late_node = list(cat_node.users.keys())[0]
            assert late_node.target == torch.quantize_per_tensor
            late_node.replace_all_uses_with(cat_node)
            model.graph.erase_node(late_node)

    model.graph.lint()
    model.recompile()
