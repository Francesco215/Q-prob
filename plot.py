import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Ensure the output directory exists
os.makedirs("img", exist_ok=True)

# Set seed for reproducible random parameters
np.random.seed(0)

def gumbel_pdf(x, mu=0.0, beta=1.0):
    """Calculates the Gumbel probability density function."""
    z = (x - mu) / beta
    return (1.0 / beta) * np.exp(-z - np.exp(-z))

def gumbel_cdf(x, mu=0.0, beta=1.0):
    """Calculates the Gumbel cumulative distribution function."""
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))

# Generate random parameters (location, scale)
params = [(np.random.randn(), np.random.rand()+0.3) for _ in range(10)]

# Generate x values for the plots
x = np.linspace(-6, 10, 1200)

# Calculate PDFs and CDFs for each set of parameters
pdfs = np.array([gumbel_pdf(x, mu, beta) for mu, beta in params])
cdfs = np.array([gumbel_cdf(x, mu, beta) for mu, beta in params])

# -----------------------------------------------------------------
# Ridgeline Plot (Refactored based on the Matplotlib guide)
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

# Use a positive spacing factor; smaller values create more overlap.
spacing = 0.3
y_offsets = np.arange(len(pdfs)) * spacing

# Generate a nice color theme using the 'viridis' colormap
colors = cm.viridis(np.linspace(0.1, 0.9, len(pdfs)))

# Plot each PDF curve from bottom to top
n_pdfs = len(pdfs)
for i, pdf in enumerate(pdfs):
    offset = y_offsets[i]
    # Assign z-order in pairs. A plot in front (lower i) gets a higher z-order pair.
    # This ensures its fill (zorder) correctly covers the line (zorder+1) of any plot behind it.
    z_order = 2 * (n_pdfs - i)
    ax.plot(x, pdf + offset, color=colors[i], linewidth=1.2, zorder=z_order + 1)
    ax.fill_between(x, offset, pdf + offset, alpha=1.0, color=colors[i], zorder=z_order)


# Add labels for each distribution on the y-axis using professional LaTeX
labels = [rf"$a' = a_{i}$" for i in range(len(params))]
ax.set_yticks(y_offsets)  # Position labels at the base of each curve
ax.set_yticklabels(labels)
ax.tick_params(axis='y', length=0)  # Hide the small tick marks

# Remove the top, right, and left plot borders (spines) for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Keep the bottom spine as a baseline
ax.spines['bottom'].set_color('gray')
ax.spines['bottom'].set_linewidth(0.5)

# Set x-axis properties with professional LaTeX
ax.set_xlim(x.min(), x.max())
ax.set_xlabel(r"$q(Q|s',a')$")

# Remove grid lines for a minimalist style
ax.grid(False)

# Adjust y-limits to ensure all curves are visible without being clipped
max_peak = max([p.max() for p in pdfs])
ax.set_ylim(-spacing * 0.25, y_offsets[-1] + max_peak + spacing * 0.5)

# Save the figure to the 'img' directory
out1 = "img/gumbel_ridgeline_plot.svg"
fig.savefig(out1, format="svg", bbox_inches="tight", transparent=False)
plt.close(fig)
print(f"Saved ridgeline plot to {out1}")


# -----------------------------------------------------------------
# PDF of the maximum of the variables (with gradient fill and outline)
# -----------------------------------------------------------------
prod_all_cdfs = np.prod(cdfs, axis=0)
pdf_max = np.zeros_like(x)
for i in range(len(pdfs)):
    others = np.prod([cdfs[j] for j in range(len(pdfs)) if j != i], axis=0)
    pdf_max += pdfs[i] * others

approx_area = np.trapz(pdf_max, x)

fig2, ax2 = plt.subplots(figsize=(8, 2)) # Reduced height for a sleeker look

# --- Gradient Fill Implementation ---
# 1. Create the gradient image.
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# 2. Display the image, stretching it to the plot's dimensions
im = ax2.imshow(gradient, aspect='auto', cmap=plt.get_cmap('viridis'),
                  extent=(x.min(), x.max(), 0, pdf_max.max()))

# 3. Use `fill_between` to create a polygon that will serve as a clipping mask
poly = ax2.fill_between(x, 0, pdf_max, facecolor='none')

# 4. Clip the gradient image with the polygon path
im.set_clip_path(poly.get_paths()[0], transform=ax2.transData)

# --- Gradient Outline Implementation ---
# 1. Create a set of points for the line segments: (x, y)
points = np.array([x, pdf_max]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 2. Create a LineCollection from the segments
lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(x.min(), x.max()))

# 3. Set the colors of the line segments based on the x-coordinate
lc.set_array(x)
lc.set_linewidth(1.5)

# 4. Add the colored line collection to the plot
ax2.add_collection(lc)


# --- Aesthetic Adjustments ---
# Remove the top, right, and left plot borders
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Keep the bottom spine as a baseline
ax2.spines['bottom'].set_color('gray')
ax2.spines['bottom'].set_linewidth(0.5)

# Remove y-axis ticks and labels
ax2.set_yticks([])

# Set x-axis properties and label
ax2.set_xlabel(r"$\max_{a'} Q(s',a')$")
ax2.set_xlim(x.min(), x.max())
# Add a 5% margin to the top to prevent clipping
ax2.set_ylim(bottom=0, top=pdf_max.max() * 1.05)


out2 = "img/gumbel_max_pdf.svg"
# Use a transparent background when saving for a cleaner look
fig2.savefig(out2, format="svg", bbox_inches="tight", transparent=False)
plt.close(fig2)
print(f"Saved max PDF plot to {out2}")


# -----------------------------------------------------------------
# NEW: PDF of the maximum with a SOLID ORANGE color scheme
# -----------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(8, 2))
x = np.linspace(-6, 10, 1200)

# Calculate PDFs and CDFs for each set of parameters
pdf_predicted = gumbel_pdf(x, mu=3, beta=1.2)
# --- Solid Color Implementation ---
# Use fill_between for the solid fill and a simple plot for the outline
ax3.fill_between(x, 0, pdf_predicted, color="tab:orange", alpha=1.0) # Lighter orange fill
ax3.plot(x, pdf_predicted, color="tab:orange", linewidth=1.5) # Darker orange line

# --- Aesthetic Adjustments ---
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_color('gray')
ax3.spines['bottom'].set_linewidth(0.5)
ax3.set_yticks([])
ax3.set_xlabel(r"$\max_{a'} Q(s',a')$")
ax3.set_xlim(x.min(), x.max())
ax3.set_ylim(bottom=0, top=pdf_predicted.max() * 1.05)

# --- Save the new plot ---
out3 = "img/gumbel_max_pdf_orange_solid.svg"
fig3.savefig(out3, format="svg", bbox_inches="tight", transparent=True)
plt.close(fig3)
print(f"Saved max PDF plot (solid orange) to {out3}")

