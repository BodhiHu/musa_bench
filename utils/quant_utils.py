import torch
from torch import fx
from tqdm import tqdm

class CustomedTracer(fx.Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """

    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )

def calibrate(model, dataloader, steps=30):
    for batch_i, (imgs, _, _, _) in tqdm(enumerate(dataloader)):
        imgs = imgs.float()
        imgs /= 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        _, _ = model(imgs)  # inference, loss outputs
        if batch_i >= steps:
            return

def _getattr(model, name):
    """customize getattr function to recursive get attribute for pytorch module"""
    name_list = name.split(".")
    for name in name_list:  # pylint: disable=redefined-argument-from-local
        model = getattr(model, name)
    return model

def postprocess(model: fx.GraphModule):
    """replace float add to quantized one, reduce dequantize ops"""
    if not isinstance(model, fx.GraphModule):
        return

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
