import numpy as np
import matplotlib.pyplot as plt

# -------- Aggregator core (Existing Code) --------
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
    if res.ndim == 0: return float(res)
    return res


def a_rev(x, neutral=0.5):
    a = _a_from_neutral(neutral)
    x_arr = np.asarray(x, dtype=float)
    res = np.empty_like(x_arr, dtype=float)
    res[x_arr > 0.5e3] = 1.0
    res[x_arr < -0.5e3] = 0.0
    mask = (x_arr <= 0.5e3) & (x_arr >= -0.5e3)
    if np.any(mask):
        ex = np.exp(x_arr[mask])
        s = ex / (1.0 + ex)
        res[mask] = np.power(s, 1.0 / a)
    if res.ndim == 0: return float(res)
    return res


def a_uni_norm(x, y, neutral=0.5):
    return a_rev(a_gen(x, neutral) + a_gen(y, neutral), neutral)


def a_abs_norm(x, y, absorbing=0.5):
    return a_rev(a_gen(x, absorbing) * a_gen(y, absorbing), absorbing)


# -------- Neuron definitions (Existing Tsetlin Neuron) --------
def _a_sinapse(snpin, wght, neutral=0.5):
    if np.isnan(snpin) or np.isnan(wght): return neutral
    return a_abs_norm(snpin, wght, neutral)


def _a_neuron_input(nin, int_state, absorbing=0.5):
    if np.isnan(nin) or np.isnan(int_state): return absorbing
    return a_abs_norm(nin, int_state, absorbing)


def _a_neuron_function(nin_array, neutral=0.5):
    arr = np.atleast_1d(nin_array).astype(float).ravel()
    res = neutral
    for v in arr:
        if not np.isnan(res) and not np.isnan(v):
            res = a_uni_norm(res, v, neutral)
    return res


def h_neuron(state, inputs, weights, neutral=0.5, absorbing=0.5):
    stt = a_r2b(state)
    inp = a_r2b(inputs)
    wht = a_r2b(weights)
    in_arr = np.atleast_1d(inp).astype(float).ravel()
    w_arr = np.atleast_1d(wht).astype(float).ravel()
    sz = len(in_arr)
    r = np.zeros(sz, dtype=float)
    for i in range(sz):
        r[i] = _a_sinapse(in_arr[i], w_arr[i], neutral)
        r[i] = _a_neuron_input(r[i], stt, absorbing)
    return _a_neuron_function(r, neutral)


def a_r2b(x, identity=0.5): return a_rev(x, identity)  # Helper from original code


# -------- OG Functions (NEW SECTION) --------
# Implementing OG_a,b1 from Section 6.2 of the provided PDF
# OG(x, y) = x^3*y^3 + x^2*y^2 + x*y
# This is a specific instance of (a,b)-OG functions used for convolution/neurons

def og_poly_func(w, x):
    """
    The OG function replacing multiplication.
    OG(w, x) = w^3*x^3 + w^2*x^2 + w*x
    """
    term1 = (w ** 3) * (x ** 3)
    term2 = (w ** 2) * (x ** 2)
    term3 = w * x
    return term1 + term2 + term3


def og_poly_deriv_w(w, x):
    """
    Partial derivative of OG(w, x) with respect to weight w.
    d/dw = 3*w^2*x^3 + 2*w*x^2 + x
    Used for Backpropagation.
    """
    term1 = 3 * (w ** 2) * (x ** 3)
    term2 = 2 * w * (x ** 2)
    term3 = x
    return term1 + term2 + term3


# -------- Classic MP neuron (Existing) --------
def f_sigmoid(sum_val):
    return 1.0 / (1.0 + np.exp(-sum_val))


def f_mp_forward(inputs, weights):
    w = np.atleast_1d(weights).astype(float).ravel()
    x = np.atleast_1d(inputs).astype(float).ravel()
    n = len(w)
    s = 0.0
    for i in range(n):
        wi = w[i]
        xi = x[i] if i < len(x) else np.nan
        if not np.isnan(wi) and not np.isnan(xi):
            s += wi * xi  # Classic multiplication
    return f_sigmoid(s)


# -------- OG Neuron Forward (NEW) --------
def f_og_forward(inputs, weights):
    """
    Implements the OG-Neuron structure from Fig 12(b) [cite: 1229]
    y = f( Sum( OG(xi, wi) ) + b )
    (Bias is omitted here to match the original mvn.py structure which implies bias=0 or included in inputs)
    """
    w = np.atleast_1d(weights).astype(float).ravel()
    x = np.atleast_1d(inputs).astype(float).ravel()
    n = len(w)
    s = 0.0
    for i in range(n):
        wi = w[i]
        xi = x[i] if i < len(x) else np.nan
        if not np.isnan(wi) and not np.isnan(xi):
            # Use OG function instead of multiplication
            s += og_poly_func(wi, xi)
    return f_sigmoid(s)


# -------- Forward apply (Modified) --------

def f_applying(links_weights, neurons_numbers, layers_number, inputs_arr, mode='mp'):
    neurons_numbers_max = int(np.max(neurons_numbers))
    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    net_outputs_arr = np.full((neurons_numbers[layers_number - 1],), np.nan, dtype=float)

    # layer 0
    layer = 0
    for neuron in range(neurons_numbers[layer]):
        weights_arr = links_weights[neuron, neuron, layer]
        if mode == 'og':
            neurons_outputs[neuron, layer] = f_og_forward(inputs_arr[neuron], weights_arr)
        else:
            neurons_outputs[neuron, layer] = f_mp_forward(inputs_arr[neuron], weights_arr)

    # layers 1..L-1
    for layer in range(1, layers_number):
        prev_outputs = neurons_outputs[:, layer - 1]
        for neuron in range(neurons_numbers[layer]):
            weights_arr = links_weights[neuron, :, layer]
            if mode == 'og':
                neurons_outputs[neuron, layer] = f_og_forward(prev_outputs, weights_arr)
            else:
                neurons_outputs[neuron, layer] = f_mp_forward(prev_outputs, weights_arr)

    res_arr = neurons_outputs[:, layers_number - 1]
    for neuron in range(neurons_numbers[layers_number - 1]):
        if not np.isnan(res_arr[neuron]):
            net_outputs_arr[neuron] = res_arr[neuron]
    return net_outputs_arr


def h_applying(links_weights, neurons_states, neurons_numbers, layers_number, inputs_arr, neutral=0.5, absorbing=0.5):
    # (Original Tsetlin logic unchanged)
    neurons_numbers_max = int(np.max(neurons_numbers))
    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    net_outputs_arr = np.full((neurons_numbers[layers_number - 1],), np.nan, dtype=float)

    layer = 0
    for neuron in range(neurons_numbers[layer]):
        in_val = inputs_arr[neuron]
        weights_arr = links_weights[neuron, neuron, layer]
        neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], in_val, weights_arr, neutral,
                                                  absorbing)

    for layer in range(1, layers_number):
        in_arr = neurons_outputs[:, layer - 1]
        for neuron in range(neurons_numbers[layer]):
            weights_arr = links_weights[neuron, :, layer]
            neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], in_arr, weights_arr, neutral,
                                                      absorbing)

    res_arr = neurons_outputs[:, layers_number - 1]
    for neuron in range(neurons_numbers[layers_number - 1]):
        if not np.isnan(res_arr[neuron]):
            net_outputs_arr[neuron] = res_arr[neuron]
    return net_outputs_arr


# -------- Backprop helpers --------
def f_delta(actual_out, wd):
    return actual_out * (1.0 - actual_out) * wd


def f_delta_out(actual_out, required_out):
    return (required_out - actual_out) * actual_out * (1.0 - actual_out)


# -------- Training (Modified to support OG) --------

def f_training(links_weights, neurons_numbers, layers_number, training_inputs, req_output_arr, training_epochs_number,
               learning_rate, mode='mp'):
    neurons_numbers_max = int(np.max(neurons_numbers))
    training_patterns_number = training_inputs.shape[1]

    neurons_outputs = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    neurons_errors = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    weight_increments = np.full_like(links_weights, np.nan, dtype=float)

    for _ in range(training_epochs_number):
        for pattern in range(training_patterns_number):
            # --- Forward Pass ---
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weights_arr = links_weights[neuron, neuron, layer]
                if mode == 'og':
                    neurons_outputs[neuron, layer] = f_og_forward(inputs_arr[neuron], weights_arr)
                else:
                    neurons_outputs[neuron, layer] = f_mp_forward(inputs_arr[neuron], weights_arr)

            for layer in range(1, layers_number):
                prev_inputs = neurons_outputs[:, layer - 1]
                for neuron in range(neurons_numbers[layer]):
                    weights_arr = links_weights[neuron, :, layer]
                    if mode == 'og':
                        neurons_outputs[neuron, layer] = f_og_forward(prev_inputs, weights_arr)
                    else:
                        neurons_outputs[neuron, layer] = f_mp_forward(prev_inputs, weights_arr)

            # --- Backward Pass (Error Calculation) ---
            layer = layers_number - 1
            for neuron in range(neurons_numbers[layer]):
                actual_output = neurons_outputs[neuron, layer]
                required_output = req_output_arr[neuron]
                neurons_errors[neuron, layer] = f_delta_out(actual_output, required_output)

            for layer in range(layers_number - 2, -1, -1):
                for neuron in range(neurons_numbers[layer]):
                    sum_weighted_errors = 0.0
                    for next_neuron in range(neurons_numbers[layer + 1]):
                        w = links_weights[next_neuron, neuron, layer + 1]
                        if not np.isnan(w):
                            # Note: For OG-Backprop in hidden layers, strictly speaking,
                            # we should also consider the derivative of OG w.r.t INPUT.
                            # But for simple comparison in this structure, we use the MP approximation for error propagation
                            # or we assume standard weighted sum error backprop is sufficient for "delta".
                            sum_weighted_errors += w * neurons_errors[next_neuron, layer + 1]
                    neurons_errors[neuron, layer] = f_delta(neurons_outputs[neuron, layer], sum_weighted_errors)

            # --- Weight Updates ---
            # Layer 0
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                err = neurons_errors[neuron, layer]
                x_val = inputs_arr[neuron]
                w_val = links_weights[neuron, neuron, layer]

                if mode == 'og':
                    # Gradient = error * d(Activation)/d(Net) * d(Net)/dw
                    # d(Net)/dw = d(OG(w,x))/dw
                    grad_factor = og_poly_deriv_w(w_val, x_val)
                    weight_increments[neuron, neuron, layer] = learning_rate * err * grad_factor
                else:
                    # Classic MP: d(w*x)/dw = x
                    weight_increments[neuron, neuron, layer] = learning_rate * err * x_val

            # Layers 1..L
            for layer in range(1, layers_number):
                for neuron in range(neurons_numbers[layer]):
                    for prev_neuron in range(neurons_numbers[layer - 1]):
                        err = neurons_errors[neuron, layer]
                        x_val = neurons_outputs[prev_neuron, layer - 1]
                        w_val = links_weights[neuron, prev_neuron, layer]

                        if mode == 'og':
                            grad_factor = og_poly_deriv_w(w_val, x_val)
                            weight_increments[neuron, prev_neuron, layer] = learning_rate * err * grad_factor
                        else:
                            weight_increments[neuron, prev_neuron, layer] = learning_rate * err * x_val

            # Apply increments
            layer = 0
            for neuron_i in range(neurons_numbers[layer]):
                neuron_j = neuron_i
                links_weights[neuron_i, neuron_j, layer] += weight_increments[neuron_i, neuron_j, layer]

            for layer in range(1, layers_number):
                for neuron_i in range(neurons_numbers[layer]):
                    for neuron_j in range(neurons_numbers[layer - 1]):
                        links_weights[neuron_i, neuron_j, layer] += weight_increments[neuron_i, neuron_j, layer]

    return links_weights


def h_training(links_weights, neurons_states, neurons_numbers, layers_number, training_inputs, req_output_arr,
               training_epochs_number, learning_rate):
    # (Original Tsetlin training logic unchanged)
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
                neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], inputs_arr[neuron],
                                                          weights_arr)

            for layer in range(1, layers_number):
                inputs_arr = neurons_outputs[:, layer - 1]
                for neuron in range(neurons_numbers[layer]):
                    weights_arr = links_weights[neuron, :, layer]
                    neurons_outputs[neuron, layer] = h_neuron(neurons_states[neuron, layer], inputs_arr[neuron],
                                                              weights_arr)

            # backward
            layer = layers_number - 1
            for neuron in range(neurons_numbers[layer]):
                actual_output = neurons_outputs[neuron, layer]
                required_output = req_output_arr[neuron]
                neurons_errors[neuron, layer] = f_delta_out(actual_output, required_output)

            for layer in range(layers_number - 2, -1, -1):
                for neuron in range(neurons_numbers[layer]):
                    sum_weighted_errors = 0.0
                    for next_neuron in range(neurons_numbers[layer + 1]):
                        sum_weighted_errors = 0.0
                        w = links_weights[next_neuron, neuron, layer + 1]
                        if not np.isnan(w):
                            sum_weighted_errors = sum_weighted_errors + w * neurons_errors[next_neuron, layer + 1]
                    neurons_errors[neuron, layer] = f_delta(neurons_outputs[neuron, layer], sum_weighted_errors)

            # weights
            layer = 0
            inputs_arr = training_inputs[:, pattern]
            for neuron in range(neurons_numbers[layer]):
                weight_increments[neuron, neuron, layer] = learning_rate * neurons_errors[neuron, layer] * inputs_arr[
                    neuron]

            for layer in range(1, layers_number):
                for neuron in range(neurons_numbers[layer]):
                    for prev_neuron in range(neurons_numbers[layer - 1]):
                        weight_increments[neuron, prev_neuron, layer] = (
                                learning_rate * neurons_errors[neuron, layer] * neurons_outputs[prev_neuron, layer - 1]
                        )

            # update
            layer = 0
            for neuron_i in range(neurons_numbers[layer]):
                neuron_j = neuron_i
                links_weights[neuron_i, neuron_j, layer] += weight_increments[neuron_i, neuron_j, layer]

            for layer in range(1, layers_number):
                for neuron_i in range(neurons_numbers[layer]):
                    for neuron_j in range(neurons_numbers[layer - 1]):
                        links_weights[neuron_i, neuron_j, layer] += weight_increments[neuron_i, neuron_j, layer]

    return links_weights


# -------- Experiment (6-neuron) with OG added --------

def run_experiment():
    neutral = 0.5
    absorbing = 0.5

    layers_number = 3
    neurons_numbers = [3, 2, 1]
    neurons_numbers_max = int(np.max(neurons_numbers))

    neurons_states = np.full((neurons_numbers_max, layers_number), np.nan, dtype=float)
    for layer in range(layers_number):
        for neuron in range(neurons_numbers[layer]):
            neurons_states[neuron, layer] = 1.0

    links_weights = np.full((neurons_numbers_max, neurons_numbers_max, layers_number), np.nan, dtype=float)

    # Initialize Weights
    layer = 0
    for net_input in range(neurons_numbers[0]):
        neuron = net_input
        links_weights[net_input, neuron, layer] = np.random.rand()

    for layer in range(1, layers_number):
        for neuron_i in range(neurons_numbers[layer]):
            for neuron_j in range(neurons_numbers[layer - 1]):
                links_weights[neuron_i, neuron_j, layer] = np.random.rand()

    learning_rate = 0.1
    training_epochs_number = 1000
    training_patterns_number = 1

    req_output_arr0 = np.array([0.0], dtype=float)
    req_output_arr1 = np.array([1.0], dtype=float)
    T = 100

    # Data collection arrays
    RF0 = np.zeros(T, dtype=float)  # MP result z*=0
    RH0 = np.zeros(T, dtype=float)  # Tsetlin result z*=0
    ROG0 = np.zeros(T, dtype=float)  # OG result z*=0
    RR0 = np.zeros(T, dtype=float) + req_output_arr0[0]

    RF1 = np.zeros(T, dtype=float)  # MP result z*=1
    RH1 = np.zeros(T, dtype=float)  # Tsetlin result z*=1
    ROG1 = np.zeros(T, dtype=float)  # OG result z*=1
    RR1 = np.zeros(T, dtype=float) + req_output_arr1[0]

    for t in range(T):
        training_inputs = np.random.rand(neurons_numbers[0], training_patterns_number)

        # --- Target = 0 ---
        lw_f = links_weights.copy()
        lw_h = links_weights.copy()
        lw_og = links_weights.copy()

        # Train
        lw_f = f_training(lw_f, neurons_numbers, layers_number, training_inputs, req_output_arr0,
                          training_epochs_number, learning_rate, mode='mp')
        lw_h = h_training(lw_h, neurons_states, neurons_numbers, layers_number, training_inputs, req_output_arr0,
                          training_epochs_number, learning_rate)
        lw_og = f_training(lw_og, neurons_numbers, layers_number, training_inputs, req_output_arr0,
                           training_epochs_number, learning_rate, mode='og')

        # Apply
        inputs_arr = training_inputs[:, 0]
        out_f = f_applying(lw_f, neurons_numbers, layers_number, inputs_arr, mode='mp')
        out_h = h_applying(lw_h, neurons_states, neurons_numbers, layers_number, inputs_arr, neutral, absorbing)
        out_og = f_applying(lw_og, neurons_numbers, layers_number, inputs_arr, mode='og')

        RF0[t] = out_f[0]
        RH0[t] = out_h[0]
        ROG0[t] = out_og[0]

        # --- Target = 1 ---
        lw_f = links_weights.copy()
        lw_h = links_weights.copy()
        lw_og = links_weights.copy()

        # Train
        lw_f = f_training(lw_f, neurons_numbers, layers_number, training_inputs, req_output_arr1,
                          training_epochs_number, learning_rate, mode='mp')
        lw_h = h_training(lw_h, neurons_states, neurons_numbers, layers_number, training_inputs, req_output_arr1,
                          training_epochs_number, learning_rate)
        lw_og = f_training(lw_og, neurons_numbers, layers_number, training_inputs, req_output_arr1,
                           training_epochs_number, learning_rate, mode='og')

        # Apply
        inputs_arr = training_inputs[:, 0]
        out_f = f_applying(lw_f, neurons_numbers, layers_number, inputs_arr, mode='mp')
        out_h = h_applying(lw_h, neurons_states, neurons_numbers, layers_number, inputs_arr, neutral, absorbing)
        out_og = f_applying(lw_og, neurons_numbers, layers_number, inputs_arr, mode='og')

        RF1[t] = out_f[0]
        RH1[t] = out_h[0]
        ROG1[t] = out_og[0]

    # --- Statistics ---
    def _summarize(arr):
        return float(np.min(arr)), float(np.max(arr)), float(np.mean(arr))

    mp0 = _summarize(RF0)
    ts0 = _summarize(RH0)
    og0 = _summarize(ROG0)

    mp1 = _summarize(RF1)
    ts1 = _summarize(RH1)
    og1 = _summarize(ROG1)

    print("\nTable 1. Parameters of the outputs")
    print("Required output z* = 0")
    print(f"  McCalloch-Pitts  min={mp0[0]:.4f}  max={mp0[1]:.4f}  avg={mp0[2]:.4f}")
    print(f"  Tsetlin          min={ts0[0]:.4f}  max={ts0[1]:.4f}  avg={ts0[2]:.4f}")
    print(f"  OG-Neuron        min={og0[0]:.4f}  max={og0[1]:.4f}  avg={og0[2]:.4f}")
    print("Required output z* = 1")
    print(f"  McCalloch-Pitts  min={mp1[0]:.4f}  max={mp1[1]:.4f}  avg={mp1[2]:.4f}")
    print(f"  Tsetlin          min={ts1[0]:.4f}  max={ts1[1]:.4f}  avg={ts1[2]:.4f}")
    print(f"  OG-Neuron        min={og1[0]:.4f}  max={og1[1]:.4f}  avg={og1[2]:.4f}")

    # --- Plotting ---
    fSz = 12

    # Figure 1: z* = 0
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(np.arange(1, T + 1), RF0, ':', color='k', linewidth=2, label='McCalloch-Pitts')
    plt.plot(np.arange(1, T + 1), RH0, '--', color='k', linewidth=2, label='Tsetlin (Fuzzy)')
    plt.plot(np.arange(1, T + 1), ROG0, '-', color='red', linewidth=2, label='OG-Neuron')
    plt.plot(np.arange(1, T + 1), RR0, '-', color='gray', linewidth=1, label='Target')
    plt.title(f'Output Convergence (Target = 0)')
    plt.xlabel('Time t')
    plt.ylabel('Output')
    plt.legend(loc='upper right')
    plt.ylim(-0.1, 0.5)

    # Figure 2: z* = 1
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(np.arange(1, T + 1), RF1, ':', color='k', linewidth=2, label='McCalloch-Pitts')
    plt.plot(np.arange(1, T + 1), RH1, '--', color='k', linewidth=2, label='Tsetlin (Fuzzy)')
    plt.plot(np.arange(1, T + 1), ROG1, '-', color='red', linewidth=2, label='OG-Neuron (New)')
    plt.plot(np.arange(1, T + 1), RR1, '-', color='gray', linewidth=1, label='Target')
    plt.title(f'Output Convergence (Target = 1)')
    plt.xlabel('Time t')
    plt.ylabel('Output')
    plt.legend(loc='lower right')
    plt.ylim(0.5, 1.1)

    plt.show()


if __name__ == "__main__":
    run_experiment()