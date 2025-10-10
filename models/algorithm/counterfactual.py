
import pandas as pd
import numpy as np
import dice_ml
import torch
from model_wrapper import MultiTaskModelWrapper
from tqdm import trange
def get_feature_baseline(inputs, targets, model, columns):
    inputs = inputs.clone().cpu().detach().numpy()
    targets = [i.clone().cpu().detach().numpy() for i in targets]
    # df_f = pd.DataFrame(inputs, columns=["stage", "age", "in.subcohort", "instit_2", "histol_2", "study_4"])
    df_f = pd.DataFrame(inputs, columns=columns)
    targets_array = np.array([t.flatten() for t in targets]).T
    # counterfactual for jth time interval not all of them! we don't need to create targets column for all time intervals
    df_p = pd.DataFrame(targets_array, columns=['targets'] * targets_array.shape[1])
    concatenated_df = pd.concat([df_f, df_p], axis=1)
    d = dice_ml.Data(dataframe=concatenated_df, continuous_features=["Age"], outcome_name='targets')
    m = dice_ml.Model(model=model, backend='PYT', model_type='classifier')
    exp = dice_ml.Dice(d, m, method="gradient")
    e1 = exp.generate_counterfactuals(df_f, total_CFs=1, desired_class="opposite", verbose=False)
    ret = np.array([example.final_cfs_df.values.tolist() for example in e1.cf_examples_list])
    return ret

# def get_attribution(model, inputs, targets):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     integrated_grad = torch.zeros(inputs.shape[0], len(targets), inputs.shape[1])
#     baseline_all = get_feature_baseline(inputs, targets, model)
#     baseline_all = torch.tensor(baseline_all)
#     print(baseline_all.shape)
#     print(inputs[0].shape)
#     # assert baseline_all.shape == inputs.shape
#     for sample_id in range(inputs.shape[0]):
#         baseline = baseline_all[sample_id][0][0:6] # 0-6 -> 7 time interval
#         baseline = baseline.to(device)
#         baseline.requires_grad = True
#         # baseline.shape == inputs[sample_id].shape
#         for task_id in range(len(targets)):
#             steps = 50
#             for i in range(inputs.shape[1]):
#                 single_feature_data = inputs[sample_id, i].unsqueeze(0)
#                 full_input = inputs[sample_id].unsqueeze(0).clone()
#                 baseline_feature_data = baseline[i].unsqueeze(0)
#
#                 full_input[:, i] = baseline_feature_data
#                 integrated_grad_for_single_feature = 0
#                 for step in range(steps):
#                     step_input = full_input.clone()
#                     step_input[:, i] = baseline_feature_data + (step / steps) * (single_feature_data - baseline_feature_data)
#
#                     model.zero_grad()
#                     step_output = model(step_input)
#                     grad = torch.autograd.grad(step_output[task_id], step_input)[0]
#                     integrated_grad_for_single_feature += grad[:, i].mean().item()
#                 integrated_grad_for_single_feature *= (single_feature_data.item() - baseline_feature_data.item()) / steps
#                 integrated_grad[sample_id, task_id, i] = integrated_grad_for_single_feature
#     model.train()
#     return integrated_grad


# def get_attribution1(model, X_train, X_columns, num_intervals, task_specific_dataset, steps=50):
#     model.eval()
#     X_train_df_temp = pd.DataFrame(torch.clone(X_train).cpu().detach().numpy(), columns=X_columns)
#     integrated_grad = torch.zeros(X_train.shape[0],  X_train.shape[1], num_intervals)
#     for task_id in range(1, num_intervals):
#         task_model_wrapper = MultiTaskModelWrapper(model, task_index=task_id)
#         dice_model = dice_ml.Model(model=task_model_wrapper.to('cpu'), backend="PYT")
#         exp = dice_ml.Dice(task_specific_dataset[task_id], dice_model, method="random")
#         cf = exp.generate_counterfactuals(X_train_df_temp, total_CFs=1,
#                                           desired_class="opposite")
#         # iterate and extract each cf for each input sample
#         cf_dfs = []
#         for cf_example in cf.cf_examples_list:
#             # Extract the final counterfactual examples as a DataFrame
#             cf_dfs.append(cf_example.final_cfs_df)
#         # Concatenate all counterfactual DataFrames into one
#         all_cf_df = pd.concat(cf_dfs, axis=0)
#         # Ensure the DataFrame has the same columns as the input data
#         all_cf_df = all_cf_df[X_columns]
#         # Convert to PyTorch tensor
#         cf_task_tensor = torch.tensor(all_cf_df.values, dtype=torch.float32, device=X_train.device)
#         task_model_wrapper.to(X_train.device)
#
#         # Precompute constants and shape dimensions
#         steps_tensor = torch.arange(0, steps, device=X_train.device) / steps  # Shape: (steps,)
#         steps_tensor = steps_tensor.view(-1, 1, 1)  # Reshape for broadcasting (50, 1, 1)
#         delta = (X_train - cf_task_tensor).unsqueeze(0)  # Shape: (1, batch_size, num_features)
#         input_cf_repeated = cf_task_tensor.unsqueeze(0)  # Shape: (1, batch_size, num_features)
#         # Create interpolated inputs for all steps in one go
#         interpolated_inputs = input_cf_repeated + steps_tensor * delta  # Shape: [50, 64, 127]
#         # print(interpolated_inputs.shape)
#         interpolated_inputs = torch.moveaxis(interpolated_inputs, 1, 0)  # Shape: [64, 50, 127]
#         for f_i in range(X_train.shape[1]):
#             X_temp = X_train.clone().repeat((steps, 1, 1))
#             X_temp = torch.moveaxis(X_temp, 1, 0).contiguous()  # Shape: [64, 50, 127]
#             X_temp[:, :, f_i] = interpolated_inputs[:, :, f_i]
#             X_temp = X_temp.view(-1, X_train.shape[1])
#             task_outputs = task_model_wrapper(X_temp)
#             grad_outputs = torch.ones_like(task_outputs)
#             grad = torch.autograd.grad(task_outputs, X_temp, grad_outputs=grad_outputs)[0]  # [3200, 127]
#             # X_temp = X_temp.view(X_train.shape[0], steps, X_train.shape[1])
#             grad = grad.view(X_train.shape[0], steps, X_train.shape[1])
#             grad = grad.sum(axis=1)
#             # print(grads.shape)
#             grads = grad[:, f_i]  # .item()
#             # print(grads)
#             grads = torch.mul(grads, (X_train[:, f_i] - cf_task_tensor[:, f_i]) / steps)
#             # print(grads)
#             integrated_grad[:, f_i, task_id] = grads
#
#     model.train()
#     return integrated_grad # for task 0 everything is 0




def generate_counterfactuals(model, X_train, X_columns, num_intervals, task_specific_dataset):
    model.eval()
    X_train_df_temp = pd.DataFrame(torch.clone(X_train).cpu().detach().numpy(), columns=X_columns)
    task_cf_dict = {}

    for task_id in range(1, num_intervals):
        task_model_wrapper = MultiTaskModelWrapper(model, task_index=task_id)
        dice_model = dice_ml.Model(model=task_model_wrapper.to('cpu'), backend="PYT")
        exp = dice_ml.Dice(task_specific_dataset[task_id], dice_model, method="random")
        cf_dfs = []
        successful_idx = []
        failed_idx = []
        for i in trange(X_train_df_temp.shape[0]):
            try:
                cf = exp.generate_counterfactuals(X_train_df_temp[i: i+1], total_CFs=1,
                                                  desired_class="opposite")
                # iterate and extract each cf for each input sample
                for cf_example in cf.cf_examples_list:
                    # Extract the final counterfactual examples as a DataFrame
                    cf_dfs.append(cf_example.final_cfs_df)
                successful_idx.append(i)
            except:
                failed_idx.append(i)
        if not cf_dfs:
            return None # if no cf example generated, return none.
        # Concatenate all counterfactual DataFrames into one
        all_cf_df = pd.concat(cf_dfs, axis=0)
        # Ensure the DataFrame has the same columns as the input data
        all_cf_df = all_cf_df[X_columns]
        task_cf_dict[task_id] = (all_cf_df, successful_idx, failed_idx)

    return task_cf_dict

def get_attribution(model, X_train, num_intervals, task_cf_dict, idx, steps=50):
    model.eval()
    integrated_grad = torch.zeros(X_train.shape[0], X_train.shape[1], num_intervals)
    for task_id in range(1, num_intervals):
        cf_df = task_cf_dict[task_id].iloc[idx]
        cf_task_tensor = torch.tensor(cf_df.values, dtype=torch.float32, device=X_train.device)
        task_model_wrapper = MultiTaskModelWrapper(model, task_index=task_id)
        task_model_wrapper.to(X_train.device)
        # Precompute constants and shape dimensions
        steps_tensor = torch.arange(0, steps, device=X_train.device) / steps  # Shape: (steps,)
        steps_tensor = steps_tensor.view(-1, 1, 1)  # Reshape for broadcasting (50, 1, 1)
        delta = (X_train - cf_task_tensor).unsqueeze(0)  # Shape: (1, batch_size, num_features)
        input_cf_repeated = cf_task_tensor.unsqueeze(0)  # Shape: (1, batch_size, num_features)
        # Create interpolated inputs for all steps in one go
        interpolated_inputs = input_cf_repeated + steps_tensor * delta  # Shape: [50, 64, 127]
        # print(interpolated_inputs.shape)
        interpolated_inputs = torch.moveaxis(interpolated_inputs, 1, 0)  # Shape: [64, 50, 127]
        for f_i in range(X_train.shape[1]):
            X_temp = X_train.clone().repeat((steps, 1, 1))
            X_temp = torch.moveaxis(X_temp, 1, 0).contiguous()  # Shape: [64, 50, 127]
            X_temp[:, :, f_i] = interpolated_inputs[:, :, f_i]
            X_temp = X_temp.view(-1, X_train.shape[1])
            task_outputs = task_model_wrapper(X_temp)
            grad_outputs = torch.ones_like(task_outputs)
            grad = torch.autograd.grad(task_outputs, X_temp, grad_outputs=grad_outputs)[0]  # [3200, 127]
            # X_temp = X_temp.view(X_train.shape[0], steps, X_train.shape[1])
            grad = grad.view(X_train.shape[0], steps, X_train.shape[1])
            grad = grad.sum(axis=1)
            # print(grads.shape)
            grads = grad[:, f_i]  # .item()
            # print(grads)
            grads = torch.mul(grads, (X_train[:, f_i] - cf_task_tensor[:, f_i]) / steps)
            # print(grads)
            integrated_grad[:, f_i, task_id] = grads

    # model.train()
    return integrated_grad



def get_attribution1(model, X_train, X_columns, num_intervals, task_specific_dataset, idx, steps=50):
    model.eval()
    X_train_df_temp = pd.DataFrame(torch.clone(X_train).cpu().detach().numpy(), columns=X_columns)
    integrated_grad = torch.zeros(X_train.shape[0],  X_train.shape[1], num_intervals)
    for task_id in range(1, num_intervals):
        task_model_wrapper = MultiTaskModelWrapper(model, task_index=task_id)
        dice_model = dice_ml.Model(model=task_model_wrapper.to('cpu'), backend="PYT")
        exp = dice_ml.Dice(task_specific_dataset[task_id], dice_model, method="random")
        cf_dfs = []
        successful_idx = []
        failed_idx = []
        for i in range(X_train_df_temp.shape[0]):
            try:
                cf = exp.generate_counterfactuals(X_train_df_temp, total_CFs=1,
                                                  desired_class="opposite")
                # iterate and extract each cf for each input sample
                for cf_example in cf.cf_examples_list:
                    # Extract the final counterfactual examples as a DataFrame
                    cf_dfs.append(cf_example.final_cfs_df)
                successful_idx.append(idx[i])
            except:
                failed_idx.append(idx[i])

        # Concatenate all counterfactual DataFrames into one
        all_cf_df = pd.concat(cf_dfs, axis=0)
        # Ensure the DataFrame has the same columns as the input data
        all_cf_df = all_cf_df[X_columns]
        # Convert to PyTorch tensor
        cf_task_tensor = torch.tensor(all_cf_df.values, dtype=torch.float32, device=X_train.device)
        task_model_wrapper.to(X_train.device)

        # Precompute constants and shape dimensions
        steps_tensor = torch.arange(0, steps, device=X_train.device) / steps  # Shape: (steps,)
        steps_tensor = steps_tensor.view(-1, 1, 1)  # Reshape for broadcasting (50, 1, 1)
        delta = (X_train - cf_task_tensor).unsqueeze(0)  # Shape: (1, batch_size, num_features)
        input_cf_repeated = cf_task_tensor.unsqueeze(0)  # Shape: (1, batch_size, num_features)
        # Create interpolated inputs for all steps in one go
        interpolated_inputs = input_cf_repeated + steps_tensor * delta  # Shape: [50, 64, 127]
        # print(interpolated_inputs.shape)
        interpolated_inputs = torch.moveaxis(interpolated_inputs, 1, 0)  # Shape: [64, 50, 127]
        for f_i in range(X_train.shape[1]):
            X_temp = X_train.clone().repeat((steps, 1, 1))
            X_temp = torch.moveaxis(X_temp, 1, 0).contiguous()  # Shape: [64, 50, 127]
            X_temp[:, :, f_i] = interpolated_inputs[:, :, f_i]
            X_temp = X_temp.view(-1, X_train.shape[1])
            task_outputs = task_model_wrapper(X_temp)
            grad_outputs = torch.ones_like(task_outputs)
            grad = torch.autograd.grad(task_outputs, X_temp, grad_outputs=grad_outputs)[0]  # [3200, 127]
            # X_temp = X_temp.view(X_train.shape[0], steps, X_train.shape[1])
            grad = grad.view(X_train.shape[0], steps, X_train.shape[1])
            grad = grad.sum(axis=1)
            # print(grads.shape)
            grads = grad[:, f_i]  # .item()
            # print(grads)
            grads = torch.mul(grads, (X_train[:, f_i] - cf_task_tensor[:, f_i]) / steps)
            # print(grads)
            integrated_grad[:, f_i, task_id] = grads

    model.train()
    return integrated_grad # for task 0 everything is 0