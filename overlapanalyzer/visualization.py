import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from openfermion.linalg import get_sparse_operator
from openfermion import load_operator
from scipy.sparse import diags
from scipy.sparse import csr_matrix, csc_matrix
from overlapanalyzer.read_ham import find_files, quick_load
from overlapanalyzer.polynomial_estimates import chebval, an_list

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "Computer Modern Roman",
    "font.size": 16,
})

def set_diag_to_zero(sparse_matrix):
    """
    Takes a sparse matrix and returns a new sparse matrix with diagonal elements set to 0.

    Args:
    sparse_matrix (csr_matrix): A scipy CSR sparse matrix.

    Returns:
    csr_matrix: A new sparse matrix with diagonal elements set to 0.
    """
    if not isinstance(sparse_matrix, (csr_matrix, csc_matrix)):
        raise ValueError("Input must be a CSR/CSC sparse matrix")

    # Extract the diagonal elements
    diagonal = sparse_matrix.diagonal()

    # Create a sparse matrix of the diagonal
    diagonal_matrix = diags(diagonal, 0, shape=sparse_matrix.shape)

    # Subtract the diagonal matrix from the original matrix
    new_matrix = sparse_matrix - diagonal_matrix
    return new_matrix


def visualize_complex_matrix(matrix, save_path):
    """
    Visualizes a given matrix using a colormap.

    Args:
    matrix (np.array): A 2D numpy array representing the matrix to be visualized.

    Returns:
    None
    """
    magnitude_matrix = np.abs(matrix)
    plt.imshow(magnitude_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    # plt.title("Matrix Visualization")
    plt.savefig(save_path, format='pdf')
    # plt.show()
    plt.close()


def visualize_chebyshev(Lower, Upper, E0, E1, Ec=None, chebyshev_degree=10):
    x_values = np.linspace(-1, 1, 400)
    plt.figure(figsize=(10, 8))
    if Ec is None:
        Ec = (E0+E1)/2
    x0 = E_to_x(E0, Lower, Upper)
    x1 = E_to_x(E1, Lower, Upper)
    xc = E_to_x(Ec, Lower, Upper)
    cheb_coeffs = an_list(xc, chebyshev_degree)
    for n in range(1, chebyshev_degree):  # Starting from 1 because cheb_coeffs[:0] would be an empty slice
        y_values = chebval(x_values, cheb_coeffs[:n])
        plt.plot(x_values, y_values, label=f'n = {n-1}')
    plt.axvline(x=x0, color='r', linestyle=':', label='x0')
    plt.axvline(x=xc, color='g', linestyle=':', label='xc')
    plt.axvline(x=x1, color='b', linestyle=':', label='x1')

    plt.title('Chebyshev Approximation of Step Function')
    plt.xlabel('x')
    # plt.ylabel('Chebval(x, cheb_coeffs[:n])')
    plt.legend()
    plt.grid(True)
    # plt.savefig("Chebyshev_polynomials.pdf")
    plt.show()


def plot_PolyDeg_vs_LB_multi(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1])
    fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
    # ax = plt.gca()
    legend_elements = [Line2D([0], [0], color='grey', lw=1.5),
                       Line2D([0], [0], color='grey', linestyle='-.', lw=1.5),
                       Line2D([0], [0], color='grey', linestyle=':', lw=1.5)]

    for i, key in enumerate(data.keys()):
        if i == 0:
            current_lowest_Eckart = data[key]['Eckart']
        color = ax1._get_lines._cycler_items[i]['color']
        x_data = np.array([i for i in range(1, len(data[key]['all_Overlap_LBs'])+1)])
        ax1.plot(x_data, data[key]['all_Overlap_LBs'], marker='*', color=color, label='Lower Bound')
        ax1.axhline(y=data[key]['exact_Overlap'], linestyle='-.', color=color, label=r'$P_{exact}$')
        ax1.axhline(y=data[key]['Eckart'], linestyle=':', color=color, label='Eckart')
        ax2.plot(x_data, data[key]['all_Overlap_LBs'], marker='*', color=color, label=f'{key} LBs')
        ax2.axhline(y=data[key]['exact_Overlap'], linestyle='-.', color=color, label=f'{key}'+r' $P_{exact}$')
        ax2.axhline(y=data[key]['Eckart'], linestyle=':', color=color, label=f'{key} Eckart')
        if data[key]['Eckart'] < current_lowest_Eckart:
            current_lowest_Eckart = data[key]['Eckart']
    
    ax1.set_ylim(-0.05, 1)
    ax2.set_ylim(current_lowest_Eckart-0.02, current_lowest_Eckart+0.02)
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .3  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    plt.xlabel('Polynomial degree')
    ax1.set_ylabel('Overlap value')
    ax1.legend(legend_elements, ['Lower Bound', r'$P_{exact}$', 'Eckart Bound'])
    plt.show()

def plot_Bounds_vs_PolyDeg_w_inset(data_dict_list, exactCons_data_dict_list, MaxDegree_list):
    # Font size variables
    label_fontsize = 18
    molecule_fontsize = 18
    label_padding = 3  # Padding for y-axis labels

    fig, axs = plt.subplots(len(data_dict_list), 2, figsize=(9, 6))

    # Increase spacing between subplots and reserve space for the legend on the right
    fig.subplots_adjust(wspace=0.4, right=0.78)  

    # Molecule names for labeling subplots
    molecule_names = [r'H$_4$', r'H$_4$', r'H$_2$O', r'H$_2$O']

    lines = []
    labels = []

    for row, data_dict in enumerate(data_dict_list):
        exactCons_data_dict = exactCons_data_dict_list[row]
        
        for col in [0, 1]:
            max_deg = MaxDegree_list[row][col]
            x_data = np.array([i for i in range(1, max_deg)])
            axs[row, col].xaxis.set_major_locator(MaxNLocator(integer=True))

            # Plot red markers (eigenvalue bounds)
            data1 = np.array(data_dict[str(col)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data1) 
            line1, = axs[row, col].plot(x_data[mask], data1[mask], marker='.', linestyle=':', label='Lower Bound (Approx.)', color='red')
            data2 = np.array(data_dict[str(col)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data2) 
            line2, = axs[row, col].plot(x_data[mask], data2[mask], marker='^', linestyle='--', label='Upper Bound (Approx.)', color='red')

            # Plot green markers (exact eigenvalues as constraints)
            data3 = np.array(exactCons_data_dict[str(col)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data3)
            line3, = axs[row, col].plot(x_data[mask], data3[mask], marker='.', linestyle=':', label='Lower Bound (Exact)', color='green')
            data4 = np.array(exactCons_data_dict[str(col)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data4)
            line4, = axs[row, col].plot(x_data[mask], data4[mask], marker='^', linestyle='--', label='Upper Bound (Exact)', color='green')

            # Plot blue horizontal line (exact overlap)
            line5 = axs[row, col].axhline(y=data_dict[str(col)]['exact_ovlp'], linestyle='-', label='Exact Overlap', color='blue')

            # Set subplot limits
            axs[row, col].set_ylim(-0.05, 1.0)
            axs[row, col].set_yticks([0.0, 0.5, 1.0])
            axs[row, col].set_xlim(0, max_deg)

            # Add molecule label to top right corner
            axs[row, col].text(
                0.95, 0.95, molecule_names[row * 2 + col],
                transform=axs[row, col].transAxes,
                fontsize=molecule_fontsize,
                verticalalignment='top', horizontalalignment='right'
            )

            # Collect legend handles and labels only once
            if row == 0 and col == 1:  # Legend belongs to the top-right plot
                lines.extend([line1, line2, line3, line4, line5])
                labels.extend([line.get_label() for line in lines])

            # Add insets for the right-side plots
            if col == 1:
                inset = inset_axes(axs[row, col], width="40%", height="35%", loc="lower right", bbox_to_anchor=(0.0, 0.35,1,1), bbox_transform=axs[row, col].transAxes)
                inset.xaxis.set_major_locator(MaxNLocator(integer=True))

                # Select the inset x-axis range
                inset_x_min = 5 if row == 0 else 6  # H_4: 5-15, H_2O: 6-24
                inset_x_max = 20 if row == 0 else 21
                # Reduce number of x-axis ticks
                inset.xaxis.set_major_locator(MaxNLocator(nbins=2))  # Adjust the number of ticks

                inset_x_data = np.array([i for i in range(inset_x_min, inset_x_max)])
                
                # Re-plot data in the inset within the selected range
                inset.plot(inset_x_data, data1[inset_x_min-1:inset_x_max-1], marker='.', linestyle=':', color='red')
                inset.plot(inset_x_data, data2[inset_x_min-1:inset_x_max-1], marker='^', linestyle='--', color='red')
                inset.plot(inset_x_data, data3[inset_x_min-1:inset_x_max-1], marker='.', linestyle=':', color='green')
                inset.plot(inset_x_data, data4[inset_x_min-1:inset_x_max-1], marker='^', linestyle='--', color='green')
                inset.axhline(y=data_dict[str(col)]['exact_ovlp'], linestyle='-', color='blue')

                # Adjust y-limits for better visibility
                inset.set_xlim(inset_x_min, inset_x_max)
                if row == 0:
                    inset.set_ylim(-0.01, 0.05)
                else:
                    inset.set_ylim(-0.02, 0.15)

    # Y-axis labels with increased padding
    axs[0, 0].set_ylabel(r'$P_0$', fontsize=label_fontsize, labelpad=label_padding)
    axs[1, 0].set_ylabel(r'$P_0$', fontsize=label_fontsize, labelpad=label_padding)
    axs[0, 1].set_ylabel(r'$P_1$', fontsize=label_fontsize, labelpad=label_padding)
    axs[1, 1].set_ylabel(r'$P_1+P_2$', fontsize=label_fontsize, labelpad=label_padding)

    # X-axis labels
    axs[1, 0].set_xlabel('Polynomial Degree', fontsize=label_fontsize)
    axs[1, 1].set_xlabel('Polynomial Degree', fontsize=label_fontsize)

    # Force y-axis tick labels to show on all plots
    for ax in axs.flat:
        ax.tick_params(axis='y', which='both', labelleft=True)

    # Place legend to the right of the top-right plot
    axs[0, 1].legend(lines, labels, loc='upper left', fontsize=10, bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout(pad=1.02)
    plt.show()


def plot_Bounds_vs_PolyDeg(data):
    for key in data.keys():
        if 'Eckart' in data[key]:
            if data[key]['Eckart'] > 0:
                # ...existing code for single axis...
                x_data = np.array([i for i in range(2, len(data[key]['Overlap_LBs'])+2)])
                plt.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', label=f'Lower Bound', color='red')
                plt.plot(x_data, data[key]['Overlap_UBs'], marker='v', linestyle=':', label='Upper Bound', color='red')
                if 'exact_overlap' in data[key]:
                    plt.axhline(y=data[key]['exact_overlap'], linestyle='-', label=r' $P_{exact}$', color='blue')
                plt.axhline(y=data[key]['Eckart'], linestyle='-.', label='Eckart Bound', color='blue')
                plt.xticks(x_data)
                plt.xlabel('Polynomial degree')
                plt.ylabel('Overlap value')
                plt.legend()
                plt.show()
            else:
                # ...existing code for two axes...
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[4, 1])
                fig.subplots_adjust(hspace=0.05)
                x_data = np.array([i for i in range(2, len(data[key]['Overlap_LBs'])+2)])
                ax1.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', color='red', label='Lower Bound')
                ax1.plot(x_data, data[key]['Overlap_UBs'], marker='v', linestyle=':', color='red', label='Upper Bound')
                if 'exact_overlap' in data[key]:
                    ax1.axhline(y=data[key]['exact_overlap'], linestyle='-', color='blue', label=r'$P_{exact}$')
                ax1.axhline(y=data[key]['Eckart'], linestyle='-.', color='blue', label='Eckart Bound')
                ax2.axhline(y=data[key]['Eckart'], linestyle='-.', color='blue')
                ax1.set_ylim(-0.05, 1.05)
                ax2.set_ylim(data[key]['Eckart']-0.02, data[key]['Eckart']+0.02)
                ax1.spines.bottom.set_visible(False)
                ax2.spines.top.set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False)
                ax1.set_xticks([])
                ax1.set_xticks([], minor=True)
                ax2.xaxis.tick_bottom()
                ax2.set_xticks(x_data)
                ax2.set_xlabel('Polynomial degree')
                ax1.set_ylabel('Overlap value')
                ax1.legend()
                plt.show()
        else:
            # ...existing code for single axis when 'Eckart' not present...
            fig, ax1 = plt.subplots()
            x_data = np.array([i for i in range(2, len(data[key]['Overlap_LBs'])+2)])
            ax1.plot(x_data, data[key]['Overlap_LBs'], marker='.', linestyle=':', color='green', label='Lower Bound')
            ax1.plot(x_data, data[key]['Overlap_UBs'], marker='v', linestyle=':', color='red', label='Upper Bound')
            if 'exact_overlap' in data[key]:
                ax1.axhline(y=data[key]['exact_overlap'], linestyle='-', color='blue', label=r'$P_{exact}$')
            ax1.set_ylim(-0.05, 1.05)
            ax1.set_xticks(x_data)
            ax1.set_xlabel('Polynomial degree')
            ax1.set_ylabel('Overlap value')
            ax1.legend()
            plt.show()

def plot_Bounds_vs_PolyDeg_Multi(data_dict_list, exactCons_data_dict_list, MaxDegree_list):
    fig, axs = plt.subplots(len(data_dict_list),2)

    for row, data_dict in enumerate(data_dict_list):
        exactCons_data_dict = exactCons_data_dict_list[row]
        for key in [0, 1]:
            max_deg = MaxDegree_list[row][key]
            x_data = np.array([i for i in range(1, max_deg)])
            axs[row, key].xaxis.set_major_locator(MaxNLocator(integer=True))

            data1 = np.array(data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data1) 
            axs[row, key].plot(x_data[mask], data1[mask], marker='.', linestyle=':', label='Lower Bound', color='red')
            data2 = np.array(data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data2) 
            axs[row, key].plot(x_data[mask], data2[mask], marker='v', linestyle=':', label='Upper Bound', color='red')
            data3 = np.array(exactCons_data_dict[str(key)]['Overlap_LBs'][:max_deg-1])
            mask = ~np.isnan(data3)
            axs[row, key].plot(x_data[mask], data3[mask], marker='.', linestyle=':', label='Lower Bound', color='green')
            data4 = np.array(exactCons_data_dict[str(key)]['Overlap_UBs'][:max_deg-1])
            mask = ~np.isnan(data4)
            axs[row, key].plot(x_data[mask], data4[mask], marker='v', linestyle=':', label='Upper Bound', color='green')
            axs[row, key].axhline(y=data_dict[str(key)]['exact_ovlp'], linestyle='-', label=r' $P_{exact}$', color='blue')
            # plt.axhline(y=data_dict[str(key)]['Eckart'], linestyle='-.', label='Eckart Bound', color='blue')
            # axs[row, key].set_xticks(x_data)
            # axs[row, key].xlabel('Polynomial degree')
            # axs[row, key].ylabel('Overlap value')
            # axs[row, key].legend()
            # plt.title(molname + f', R = {key}')
    plt.tight_layout()
    plt.show()


def plot_bounds_payload(payload, ax=None, title=None, color=None, update_limits=True, **kwargs):
    """Plot lower/upper polynomial bounds for a single payload returned by OverlapEstimator.compute_bounds().

    payload: dict with keys 'mode', 'degree_results' (list of {'degree','lower','upper'}) and optional 'exact_overlap'.
    """
    import numpy as _np
    if ax is None:
        fig, ax = plt.subplots()
    degrees = []
    lowers = []
    uppers = []
    deg_results = payload.get('degree_results', [])
    # degree_results may be list of dicts or formatted differently
    for entry in deg_results:
        if isinstance(entry, dict):
            degrees.append(entry.get('degree'))
            lowers.append(entry.get('lower'))
            uppers.append(entry.get('upper'))
    if not degrees:
        return ax
    degrees = _np.array(degrees)
    lowers = _np.array(lowers)
    uppers = _np.array(uppers)

    if kwargs.get('show_monotonic', False):
        # Monotonic Lower Bounds
        eff_lowers = []
        curr_lb = -float('inf')
        for val in lowers:
            if val is not None and not _np.isnan(val):
                if val > curr_lb:
                    curr_lb = val
            eff_lowers.append(curr_lb if curr_lb != -float('inf') else _np.nan)
        lowers = _np.array(eff_lowers)

        # Monotonic Upper Bounds
        eff_uppers = []
        curr_ub = float('inf')
        for val in uppers:
            if val is not None and not _np.isnan(val):
                if val < curr_ub:
                    curr_ub = val
            eff_uppers.append(curr_ub if curr_ub != float('inf') else _np.nan)
        uppers = _np.array(eff_uppers)

    lbl = payload.get('mode', 'bounds')
    c = color or None
    ax.plot(degrees, lowers, marker='.', linestyle=':', label=f'{lbl} LB', color=c)
    ax.plot(degrees, uppers, marker='*', linestyle='--', label=f'{lbl} UB', color=c)
    if 'exact_overlap' in payload:
        ax.axhline(y=payload['exact_overlap'], color='blue', linestyle='-', linewidth=1, label=f"{lbl} exact")
    if title:
        ax.set_title(title)
    # ax.set_xlabel('Polynomial degree')
    # ax.set_ylabel('Overlap')
    
    if update_limits:
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_xlim(0, max(degrees)+1)
        if kwargs.get('x_spacing', False):
            ax.set_xticks([i for i in range(0, max(degrees)+2, kwargs.get('x_spacing', 3))])
    # ax.legend()
    return ax


def plot_bounds(results, save_path=None, figsize=None, max_deg=None, **kwargs):
    """Plot one or many payloads produced by OverlapEstimator.compute_bounds().

    results: single payload dict or list of payload dicts.
    If `save_path` is provided, the figure will be saved to that path.
    Optional `figsize` (width, height) can be provided to control figure size.
    """
    if results is None:
        raise ValueError("No results provided to plot_bounds().")
    if not isinstance(results, (list, tuple)):
        payloads = [results]
    else:
        payloads = list(results)
    # create figure with optional custom size
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    # colors = plt.rcParams.get('axes.prop_cycle').by_key().get('color', [])
    colors = ['red', 'green']
    
    global_max_degree = 0
    for i, p in enumerate(payloads):
        # Track max degree
        deg_results = p.get('degree_results', [])
        current_max = 0
        for entry in deg_results:
            if isinstance(entry, dict):
                d = entry.get('degree')
                if d is not None and d > current_max:
                    current_max = d
        if current_max > global_max_degree:
            global_max_degree = current_max

        c = colors[i % len(colors)] if colors else None
        plot_bounds_payload(p, ax=ax, title=None, color=c, update_limits=False, **kwargs)
        
    # Determine effective max degree for plotting
    final_max_degree = global_max_degree
    if max_deg is not None and max_deg < global_max_degree:
        final_max_degree = max_deg

    # Set global limits
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim(0, final_max_degree+1)
    if kwargs.get('x_spacing', False):
        ax.set_xticks([i for i in range(0, final_max_degree+2, kwargs.get('x_spacing', 3))])

    if kwargs.get('include_title', False):
        fig.suptitle('Overlap bounds vs polynomial degree')
    if kwargs.get('include_label', True):
        labels_str = kwargs.get('ylabel', '0')
        if labels_str == '0':
            plt.ylabel(r'$P_0$')
        elif labels_str == '1':
            plt.ylabel(r'$P_1$')
        elif labels_str == '1+2':
            plt.ylabel(r'$P_1 + P_2$')
        elif labels_str == 'below':
            plt.ylabel(r'$P_{<}$')
        else:
            plt.ylabel(f'$P_{{{labels_str}}}$')
    fig.subplots_adjust(left=kwargs.get('plot_left_adjust', 0.1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if kwargs.get('show', True):
        plt.show()
    plt.close(fig)

def plot_inaccuracy_vs_LB(data):
    for i, key in enumerate(data.keys()):
        inaccuracies = np.array(data[key]['inaccuracies'])*2/data[key]['gap']
        O_exact = data[key]['O_exact']
        for pol_deg in data[key]["Overlap_LBs"].keys():
            y_data = np.abs(np.array(data[key]["Overlap_LBs"][pol_deg]) - O_exact) / O_exact
            plt.semilogy(inaccuracies, y_data, marker='.', linestyle=':', label=pol_deg)
        plt.legend()
        plt.ylabel('Relative error')
        plt.xlabel('Energy Inaccuracy')
        plt.show()
        plt.close()
        O_exact = data[key]['O_exact']
        for pol_deg in data[key]["Overlap_UBs"].keys():
            y_data = np.abs(np.array(data[key]["Overlap_UBs"][pol_deg]) - O_exact) / O_exact
            plt.semilogy(inaccuracies, y_data, marker='.', linestyle=':', label=pol_deg)
        plt.legend()
        plt.ylabel('Relative error')
        plt.xlabel('Energy Inaccuracy')
        plt.show()
        plt.close()
        
if __name__ == "__main__":

    dir = 'hamiltonian_gen_test/h2o/4e4o'
    load_ham_type = 'fer'
    ham_filenames = find_files(dir+'/ham_'+load_ham_type,".data") # list of filenames ending with ".data"

    for i, filename in enumerate(ham_filenames):
        H_loaded_q = load_operator(data_directory=dir+'/ham_'+load_ham_type, file_name=filename, plain_text=True)
        H_diag_0 = set_diag_to_zero(get_sparse_operator(H_loaded_q))
        visualize_complex_matrix(H_diag_0.toarray(), save_path=dir+'/ham_'+load_ham_type+'/'+filename[:-5]+'.pdf')