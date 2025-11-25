import numpy as np
import matplotlib.pyplot as plt


# -------- Aggregator core --------

_EPS_BIG = 2.22e16
_EPS_SMALL = 2.22e-16

def _a_from_neutral(neutral: float) -> float:
    if neutral == 0:
        return _EPS_SMALL
    elif neutral == 1:
        return _EPS_BIG
    else:
        return 1.0 / np.log2(1.0 / neutral)

def a_gen(y, neutral=0.5):
    # vectorized: handle scalar or array
    a = _a_from_neutral(neutral)
    y_arr = np.asarray(y, dtype=float)
    res = np.empty_like(y_arr, dtype=float)

    mask0 = (y_arr == 0.0)
    mask1 = (y_arr == 1.0)
    maskm = ~(mask0 | mask1)

    res[mask0] = -_EPS_BIG
    res[mask1] = _EPS_BIG
    if np.any(maskm):
        ya = np.power(y_arr[maskm], a)
        res[maskm] = np.log(ya / (1.0 - ya))

    if res.ndim == 0:
        return float(res)
    return res

def a_rev(x, neutral=0.5):
    # vectorized: handle scalar or array
    a = _a_from_neutral(neutral)
    x_arr = np.asarray(x, dtype=float)
    res = np.empty_like(x_arr, dtype=float)

    res[x_arr > 0.5e3] = 1.0
    res[x_arr < -0.5e3] = 0.0
    mask = (x_arr <= 0.5e3) & (x_arr >= -0.5e3)
    if np.any(mask):
        ex = np.exp(x_arr[mask])
        s = ex / (1.0 + ex) #sigmoid
        res[mask] = np.power(s, 1.0 / a)

    if res.ndim == 0:
        return float(res)
    return res

def a_uni_norm(x, y, neutral=0.5):
    return a_rev(a_gen(x, neutral) + a_gen(y, neutral), neutral)

def a_abs_norm(x, y, absorbing=0.5):
    return a_rev(a_gen(x, absorbing) * a_gen(y, absorbing), absorbing)

def a_not(y, neutral=0.5):
    return a_rev(-a_gen(y, neutral), neutral)

def a_substract(x, y, neutral=0.5):
    return a_rev(a_gen(x, neutral) - a_gen(y, neutral), neutral)

def a_divide(x, y, absorbing=0.5):
    return a_rev(a_gen(x, absorbing) / a_gen(y, absorbing), absorbing)

def a_b2r(x, identity=0.5):
    # element-wise a_gen
    return a_gen(x, identity)

def a_r2b(x, identity=0.5):
    # element-wise a_rev
    return a_rev(x, identity)


# -------- Neuron definitions --------

def _a_sinapse(snpin, wght, neutral=0.5):
    # 若任一 NaN，返回 neutral；否则用absnorm归一化
    if np.isnan(snpin) or np.isnan(wght):
        return neutral
    return a_abs_norm(snpin, wght, neutral)

def _a_neuron_input(nin, int_state, absorbing=0.5):
    # 若任一 NaN，返回 absorbing；否则用乘性absnorm归一化
    if np.isnan(nin) or np.isnan(int_state):
        return absorbing
    return a_abs_norm(nin, int_state, absorbing)

def _a_neuron_function(nin_array, neutral=0.5):
    # 用二元和式归一化累积
    arr = np.atleast_1d(nin_array).astype(float).ravel()
    res = neutral
    for v in arr:
        if not np.isnan(res) and not np.isnan(v):
            res = a_uni_norm(res, v, neutral)
    return res

def a_neuron(state, inputs, weights, neutral=0.5, absorbing=0.5):
    in_arr = np.atleast_1d(inputs).astype(float).ravel()
    w_arr = np.atleast_1d(weights).astype(float).ravel()
    sz = len(in_arr)
    r = np.zeros(sz, dtype=float)

    # 注意：使用 inputs 的长度 sz 迭代
    for i in range(sz):
        r[i] = _a_sinapse(in_arr[i], w_arr[i], neutral)
        r[i] = _a_neuron_input(r[i], state, absorbing)

    return _a_neuron_function(r, neutral)

def h_neuron(state, inputs, weights, neutral=0.5, absorbing=0.5):
    stt = a_r2b(state)
    inp = a_r2b(inputs)
    wht = a_r2b(weights)
    return a_neuron(stt, inp, wht, neutral, absorbing)


# -------- Classic MP neuron --------

def f_sigmoid(inputs, weights):
    # 只累加非 NaN 对应项
    w = np.atleast_1d(weights).astype(float).ravel()
    x = np.atleast_1d(inputs).astype(float).ravel()
    n = len(w)
    s = 0.0
    for i in range(n):
        wi = w[i]
        xi = x[i] if i < len(x) else np.nan
        if not np.isnan(wi) and not np.isnan(xi):
            s += wi * xi
    return 1.0 / (1.0 + np.exp(-s))


# -------- Forward apply --------

def f_applying(links_weights, neurons_numbers, layers_number, inputs_arr):
    neurons_numbers_max = int(np.max(neurons_numbers))
    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    net_outputs_arr = np.full((neurons_numbers[layers_number-1],), np.nan, dtype=float)

    # layer 0
    layer = 0
    for neuron in range(neurons_numbers[layer]):
        weights_arr = links_weights[neuron, neuron, layer]
        neurons_outputs[neuron, layer] = f_sigmoid(inputs_arr[neuron], weights_arr)

    # layers 1..L-1
    for layer in range(1, layers_number):
        prev_outputs = neurons_outputs[:, layer-1]
        for neuron in range(neurons_numbers[layer]):
            weights_arr = links_weights[neuron, :, layer]
            neurons_outputs[neuron, layer] = f_sigmoid(prev_outputs, weights_arr)

    res_arr = neurons_outputs[:, layers_number-1]
    for neuron in range(neurons_numbers[layers_number-1]):
        if not np.isnan(res_arr[neuron]):
            net_outputs_arr[neuron] = res_arr[neuron]
    return net_outputs_arr

def h_applying(links_weights, neurons_states, neurons_numbers, layers_number, inputs_arr, neutral=0.5, absorbing=0.5):
    neurons_numbers_max = int(np.max(neurons_numbers))
    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    net_outputs_arr = np.full((neurons_numbers[layers_number-1],), np.nan, dtype=float)

    # layer 0
    layer = 0
    for neuron in range(neurons_numbers[layer]):
        in_val = inputs_arr[neuron]
        weights_arr = links_weights[neuron, neuron, layer]
        neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], in_val, weights_arr, neutral, absorbing)

    # layers 1..L-1 (传入整列 prev_outputs)
    for layer in range(1, layers_number):
        in_arr = neurons_outputs[:, layer-1]
        for neuron in range(neurons_numbers[layer]):
            weights_arr = links_weights[neuron, :, layer]
            neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], in_arr, weights_arr, neutral, absorbing)

    res_arr = neurons_outputs[:, layers_number-1]
    for neuron in range(neurons_numbers[layers_number-1]):
        if not np.isnan(res_arr[neuron]):
            net_outputs_arr[neuron] = res_arr[neuron]
    return net_outputs_arr


# -------- Backprop helpers --------

def f_delta(actual_out, wd):
    return actual_out * (1.0 - actual_out) * wd

def f_delta_out(actual_out, required_out):
    return (required_out - actual_out) * actual_out * (1.0 - actual_out)


# -------- Training --------

def f_training(links_weights, neurons_numbers, layers_number, training_inputs, req_output_arr, training_epochs_number, learning_rate):
    neurons_numbers_max = int(np.max(neurons_numbers))
    training_patterns_number = training_inputs.shape[1]

    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    neurons_errors = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    weight_increments = np.full_like(links_weights, np.nan, dtype=float)

    for _ in range(training_epochs_number):
        for pattern in range(training_patterns_number):
            # forward
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weights_arr = links_weights[neuron, neuron, layer]
                neurons_outputs[neuron, layer] = f_sigmoid(inputs_arr[neuron], weights_arr)

            for layer in range(1, layers_number):
                inputs_arr = neurons_outputs[:, layer-1]
                for neuron in range(neurons_numbers[layer]):
                    weights_arr = links_weights[neuron, :, layer]
                    neurons_outputs[neuron, layer] = f_sigmoid(inputs_arr, weights_arr)

            # backward (注意 sum_weighted_errors 在内层循环被反复置零的写法)
            layer = layers_number - 1
            for neuron in range(neurons_numbers[layer]):
                actual_output = neurons_outputs[neuron, layer]
                required_output = req_output_arr[neuron]
                neurons_errors[neuron, layer] = f_delta_out(actual_output, required_output)

            for layer in range(layers_number-2, -1, -1):
                for neuron in range(neurons_numbers[layer]):
                    sum_weighted_errors = 0.0
                    for next_neuron in range(neurons_numbers[layer+1]):
                        sum_weighted_errors = 0.0  # 重置在内层循环
                        w = links_weights[next_neuron, neuron, layer+1]
                        if not np.isnan(w):
                            sum_weighted_errors = sum_weighted_errors + w * neurons_errors[next_neuron, layer+1]
                    neurons_errors[neuron, layer] = f_delta(neurons_outputs[neuron, layer], sum_weighted_errors)

            # weight increments
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weight_increments[neuron, neuron, layer] = learning_rate * neurons_errors[neuron, layer] * inputs_arr[neuron]

            for layer in range(1, layers_number):
                for neuron in range(neurons_numbers[layer]):
                    for prev_neuron in range(neurons_numbers[layer-1]):
                        weight_increments[neuron, prev_neuron, layer] = (
                            learning_rate * neurons_errors[neuron, layer] * neurons_outputs[prev_neuron, layer-1]
                        )

            # update links weights
            layer = 0
            for neuron_i in range(neurons_numbers[layer]):
                neuron_j = neuron_i
                links_weights[neuron_i, neuron_j, layer] = links_weights[neuron_i, neuron_j, layer] + weight_increments[neuron_i, neuron_j, layer]

            for layer in range(1, layers_number):
                for neuron_i in range(neurons_numbers[layer]):
                    for neuron_j in range(neurons_numbers[layer-1]):
                        links_weights[neuron_i, neuron_j, layer] = links_weights[neuron_i, neuron_j, layer] + weight_increments[neuron_i, neuron_j, layer]

    return links_weights

def h_training(links_weights, neurons_states, neurons_numbers, layers_number, training_inputs, req_output_arr, training_epochs_number, learning_rate):
    neurons_numbers_max = int(np.max(neurons_numbers))
    training_patterns_number = training_inputs.shape[1]

    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    neurons_errors = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    weight_increments = np.full_like(links_weights, np.nan, dtype=float)

    for _ in range(training_epochs_number):
        for pattern in range(training_patterns_number):
            # forward
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weights_arr = links_weights[neuron, neuron, layer]
                neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], inputs_arr[neuron], weights_arr)

            # 注意：把整列 prev_outputs 赋给 inputs_arr，
            # 但传给 h_neuron 的是 inputs_arr[neuron]（单个标量），从而仅用对角输入参与训练
            for layer in range(1, layers_number):
                inputs_arr = neurons_outputs[:, layer-1]
                for neuron in range(neurons_numbers[layer]):
                    weights_arr = links_weights[neuron, :, layer]
                    neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], inputs_arr[neuron], weights_arr)

            # backward（sum_weighted_errors 内层循环重置）
            layer = layers_number - 1
            for neuron in range(neurons_numbers[layer]):
                actual_output = neurons_outputs[neuron, layer]
                required_output = req_output_arr[neuron]
                neurons_errors[neuron, layer] = f_delta_out(actual_output, required_output)

            for layer in range(layers_number-2, -1, -1):
                for neuron in range(neurons_numbers[layer]):
                    sum_weighted_errors = 0.0
                    for next_neuron in range(neurons_numbers[layer+1]):
                        sum_weighted_errors = 0.0  # MATLAB: 重置在内层循环
                        w = links_weights[next_neuron, neuron, layer+1]
                        if not np.isnan(w):
                            sum_weighted_errors = sum_weighted_errors + w * neurons_errors[next_neuron, layer+1]
                    neurons_errors[neuron, layer] = f_delta(neurons_outputs[neuron, layer], sum_weighted_errors)

            # weight increments
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weight_increments[neuron, neuron, layer] = learning_rate * neurons_errors[neuron, layer] * inputs_arr[neuron]

            for layer in range(1, layers_number):
                for neuron in range(neurons_numbers[layer]):
                    for prev_neuron in range(neurons_numbers[layer-1]):
                        weight_increments[neuron, prev_neuron, layer] = (
                            learning_rate * neurons_errors[neuron, layer] * neurons_outputs[prev_neuron, layer-1]
                        )

            # update
            layer = 0
            for neuron_i in range(neurons_numbers[layer]):
                neuron_j = neuron_i
                links_weights[neuron_i, neuron_j, layer] = links_weights[neuron_i, neuron_j, layer] + weight_increments[neuron_i, neuron_j, layer]

            for layer in range(1, layers_number):
                for neuron_i in range(neurons_numbers[layer]):
                    for neuron_j in range(neurons_numbers[layer-1]):
                        links_weights[neuron_i, neuron_j, layer] = links_weights[neuron_i, neuron_j, layer] + weight_increments[neuron_i, neuron_j, layer]

    return links_weights


# -------- Experiment (6-neuron) --------

def run_experiment():
    # identities
    neutral = 0.5
    absorbing = 0.5

    # network structure
    layers_number = 3
    neurons_numbers = [3, 2, 1]
    neurons_numbers_max = int(np.max(neurons_numbers))

    # neurons' internal states matrix: shape [max_neurons, layers]
    neurons_states = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    for layer in range(layers_number):
        for neuron in range(neurons_numbers[layer]):
            neurons_states[neuron, layer] = 1.0

    # links weights 3D tensor: [to_neuron, from_neuron, layer]
    links_weights = np.full((neurons_numbers_max, neurons_numbers_max, layers_number), np.nan, dtype=float)

    # init layer 0: diagonal only
    layer = 0
    for net_input in range(neurons_numbers[0]):
        neuron = net_input
        links_weights[net_input, neuron, layer] = np.random.rand()

    # init layers 1..L-1: fully connected (valid part)
    for layer in range(1, layers_number):
        for neuron_i in range(neurons_numbers[layer]):
            for neuron_j in range(neurons_numbers[layer-1]):
                links_weights[neuron_i, neuron_j, layer] = np.random.rand()

    learning_rate = 0.1
    training_epochs_number = 1000
    training_patterns_number = 1

    # one random input pattern per run
    req_output_arr0 = np.array([0.0], dtype=float)
    req_output_arr1 = np.array([1.0], dtype=float)
    T = 100

    RF0 = np.zeros(T, dtype=float)
    RH0 = np.zeros(T, dtype=float)
    RR0 = np.zeros(T, dtype=float) + req_output_arr0[0]

    RF1 = np.zeros(T, dtype=float)
    RH1 = np.zeros(T, dtype=float)
    RR1 = np.zeros(T, dtype=float) + req_output_arr1[0]

    for t in range(T):
        training_inputs = np.random.rand(neurons_numbers[0], training_patterns_number)

        # target = 0
        links_weights_f = links_weights.copy()
        links_weights_h = links_weights.copy()

        links_weights_f = f_training(links_weights_f, neurons_numbers, layers_number,
                                     training_inputs, req_output_arr0, training_epochs_number, learning_rate)
        links_weights_h = h_training(links_weights_h, neurons_states, neurons_numbers, layers_number,
                                     training_inputs, req_output_arr0, training_epochs_number, learning_rate)
        inputs_arr = training_inputs[:, 0]
        net_outputs_arr_f = f_applying(links_weights_f, neurons_numbers, layers_number, inputs_arr)
        net_outputs_arr_h = h_applying(links_weights_h, neurons_states, neurons_numbers, layers_number, inputs_arr,
                                       neutral=neutral, absorbing=absorbing)
        RF0[t] = net_outputs_arr_f[0]
        RH0[t] = net_outputs_arr_h[0]

        # target = 1
        links_weights_f = links_weights.copy()
        links_weights_h = links_weights.copy()

        links_weights_f = f_training(links_weights_f, neurons_numbers, layers_number,
                                     training_inputs, req_output_arr1, training_epochs_number, learning_rate)
        links_weights_h = h_training(links_weights_h, neurons_states, neurons_numbers, layers_number,
                                     training_inputs, req_output_arr1, training_epochs_number, learning_rate)
        inputs_arr = training_inputs[:, 0]
        net_outputs_arr_f = f_applying(links_weights_f, neurons_numbers, layers_number, inputs_arr)
        net_outputs_arr_h = h_applying(links_weights_h, neurons_states, neurons_numbers, layers_number, inputs_arr,
                                       neutral=neutral, absorbing=absorbing)
        RF1[t] = net_outputs_arr_f[0]
        RH1[t] = net_outputs_arr_h[0]

    # --- Table 1 stats ---
    def _summarize(arr):
        return float(np.min(arr)), float(np.max(arr)), float(np.mean(arr))

    mp0 = _summarize(RF0)  # McCalloch-Pitts, z*=0
    ts0 = _summarize(RH0)  # Tsetlin,        z*=0
    mp1 = _summarize(RF1)  # McCalloch-Pitts, z*=1
    ts1 = _summarize(RH1)  # Tsetlin,        z*=1

    print("\nTable 1. Parameters of the outputs")
    print("Required output z* = 0")
    print(f"  McCalloch-Pitts  min={mp0[0]:.4f}  max={mp0[1]:.4f}  average={mp0[2]:.4f}")
    print(f"  Tsetlin          min={ts0[0]:.4f}  max={ts0[1]:.4f}  average={ts0[2]:.4f}")
    print("Required output z* = 1")
    print(f"  McCalloch-Pitts  min={mp1[0]:.4f}  max={mp1[1]:.4f}  average={mp1[2]:.4f}")
    print(f"  Tsetlin          min={ts1[0]:.4f}  max={ts1[1]:.4f}  average={ts1[2]:.4f}")

    # --- 表格图 ---
    fig_tbl, ax_tbl = plt.subplots(figsize=(9, 2.2))
    ax_tbl.axis('off')
    col_labels = ["Model", "z*=0 min", "z*=0 max", "z*=0 avg", "z*=1 min", "z*=1 max", "z*=1 avg"]
    cell_text = [
        ["McCalloch-Pitts", f"{mp0[0]:.4f}", f"{mp0[1]:.4f}", f"{mp0[2]:.4f}",
         f"{mp1[0]:.4f}", f"{mp1[1]:.4f}", f"{mp1[2]:.4f}"],
        ["Tsetlin", f"{ts0[0]:.4f}", f"{ts0[1]:.4f}", f"{ts0[2]:.4f}",
         f"{ts1[0]:.4f}", f"{ts1[1]:.4f}", f"{ts1[2]:.4f}"],
    ]
    tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False);
    tbl.set_fontsize(12);
    tbl.scale(1, 1.4)
    fig_tbl.tight_layout()

    # plot
    fSz = 15

    plt.figure()
    plt.grid(True)
    plt.plot(np.arange(1, T+1), RF0, ':', color='k', linewidth=2, label='McCalloch-Pitts neurons')
    plt.plot(np.arange(1, T+1), RH0, '--', color='k', linewidth=2, label='Tsetlin neurons')
    plt.plot(np.arange(1, T+1), RR0, '-', color='k', linewidth=2, label='Required output')
    plt.title(f'Required output z* = {req_output_arr0[0]}')
    plt.xlabel('Time t')
    plt.ylabel('Outputs z^6(t)')
    plt.xlim(0, T+1)
    plt.ylim(-0.01, 0.1)
    plt.legend(loc='best')
    plt.gca().tick_params(labelsize=fSz)

    plt.figure()
    plt.grid(True)
    plt.plot(np.arange(1, T+1), RF1, ':', color='k', linewidth=2, label='McCalloch-Pitts neurons')
    plt.plot(np.arange(1, T+1), RH1, '--', color='k', linewidth=2, label='Tsetlin neurons')
    plt.plot(np.arange(1, T+1), RR1, '-', color='k', linewidth=2, label='Required output')
    plt.title(f'Required output z* = {req_output_arr1[0]}')
    plt.xlabel('Time t')
    plt.ylabel('Outputs z^6(t)')
    plt.xlim(0, T+1)
    plt.ylim(0.9, 1.01)
    plt.legend(loc='best')
    plt.gca().tick_params(labelsize=fSz)

    plt.show()


if __name__ == "__main__":
    run_experiment()
