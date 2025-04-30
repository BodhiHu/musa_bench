import numpy as np
from mtnn_api import MTNN, ffi
import ctypes
import torch

mtnn_lib = ffi.dlopen("/usr/lib/libmtnnrt.so")

class MTNNSession:
    def __init__(self, model_path, core_id = 0):
        self.mtnn = MTNN()
        self.mtnn.init(model_path, ffi.NULL)

        freq = mtnn_lib.MTNN_POWER_FIXED_FREQUENCY_LEVEL4
        core_info_c = ffi.new("int*", core_id)
        freq_info_c = ffi.new("int*", freq)
        self.mtnn.set_info(mtnn_lib.MTNN_SET_NPU_INDEX, core_info_c, ffi.sizeof("int"))
        self.mtnn.set_info(mtnn_lib.MTNN_SET_NPU_FREQUENCY, freq_info_c, ffi.sizeof("int"))

        """获取模型的输入输出数量"""
        self.io_num_c = ffi.new("mtnn_input_output_num *")
        self.io_num_c.n_input = 0
        self.io_num_c.n_output = 0
        self.mtnn.get_info(mtnn_lib.MTNN_GET_IN_OUT_NUM, self.io_num_c, ffi.sizeof("mtnn_input_output_num"))

        self.inputs_mem = self.mtnn.get_inputs(self.io_num_c.n_input)
        self.outputs_mem = self.mtnn.get_outputs(self.io_num_c.n_output)

        """获取输入属性"""
        self.inputs_attr = []
        num_inputs = self.get_num_inputs()
        for i in range(num_inputs):
            attr = ffi.new("mtnn_tensor_attr*")
            self.mtnn.get_info(mtnn_lib.MTNN_GET_INPUT_ATTR, attr, ffi.sizeof("mtnn_tensor_attr"))
            self.inputs_attr.append(attr)
        
        """获取输出属性"""
        self.outputs_attr = []
        num_outputs = self.get_num_outputs()
        for i in range(num_outputs):
            attr = ffi.new("mtnn_tensor_attr*")
            self.mtnn.get_info(mtnn_lib.MTNN_GET_OUTPUT_ATTR, attr, ffi.sizeof("mtnn_tensor_attr"))
            self.outputs_attr.append(attr)

    def run(self, input_feed):
        for index, data in input_feed.items():
            input_mem = self.inputs_mem[index]
            input_buffer = ffi.cast("char *", input_mem.logical_addr)
            input_size = input_mem.size
            data_ptr = ffi.cast("char *", data.ctypes.data)
            size = data.size * ffi.sizeof("float")
            assert size == input_size, f"Input index {index} size mismatch: {size} != {input_size}"
            ffi.memmove(input_buffer, data_ptr, input_mem.size)
        

        self.mtnn.set_inputs(self.io_num_c.n_input, ffi.NULL)
        self.mtnn.inference(ffi.NULL)

        outputs = []
        num_outputs = self.get_num_outputs()
        for i in range(num_outputs):
            output_mem = self.outputs_mem[i]
            output_attr = self.outputs_attr[i]
            output_data = np.frombuffer(ffi.buffer(output_mem.logical_addr, output_mem.size), dtype=np.float32)
            new_shape = tuple(output_attr.dims[0:output_attr.n_dims])
            new_shape = new_shape[::-1]
            output_data = output_data.reshape(new_shape).copy()
            outputs.append(output_data)
        
        return outputs

    def get_num_inputs(self):
        return self.io_num_c.n_input

    def get_num_outputs(self):
        return self.io_num_c.n_output

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mtnn.destroy()


class MtnnYOLOModel:
    def __init__(self, model_path, core_id:int = 0):
        self.model = MTNNSession(model_path, core_id)

    def __call__(self, input: np.ndarray):
        preds = self.model.run({0: input})
        return preds

