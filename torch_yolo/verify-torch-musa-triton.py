import torch_compat as torch;

torch._logging.set_logs(dynamo=50, inductor=50)
model = torch.nn.Linear(10, 10)
compiled_model = torch.compile(model, backend="inductor")
input = torch.randn(10, 10)
output = compiled_model(input)
print(f"\n>> backend = \n{compiled_model._compiled_model.backend}\n\n")
print(f"\n>> codegen = \n{torch._inductor.codegen}\n\n")
torch._dynamo.utils.save_code_for_debugging(compiled_model)

