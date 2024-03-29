from itertools import chain
from tensorboard.backend.event_processing import event_accumulator
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For the color palette

N_HISTOGRAMS = 10
SIMPLIFIED = False

# Step 1: Replace 'path/to/logs' with the path to your TensorBoard log files
log_file_path = 'FullyDecoder_logs/lightning_logs/beamforming_2024-03-28 20:45:32.246514/events.out.tfevents.1711655132.janis-Ubuntu18.220839.0'

# Initialize an event accumulator
ea = event_accumulator.EventAccumulator(log_file_path,
                                        size_guidance={
                                            event_accumulator.HISTOGRAMS: 0,  # Load all histograms
                                        })

# Load the events from file
ea.Reload()

# Predefined tags and their pairing
if not SIMPLIFIED:
    tags_pairs = [
        ('actual_values_permittivity', 'predicted_values_permittivity'),
        ('actual_values_conductivity', 'predicted_values_conductivity'),
        ('actual_values_sdf', 'predicted_values_sdf')
    ]
else:
    tags_pairs = [
        ('actual_values_sdf', 'predicted_values_sdf')
    ]

flattened_tags = list(chain(*tags_pairs))

# Create a color palette for the steps
colors = px.colors.qualitative.Vivid[:N_HISTOGRAMS]

# Create subplot figure
fig = make_subplots(rows=len(tags_pairs),
                    cols=2,
                    subplot_titles=[f'{title}' for title in flattened_tags],
                    vertical_spacing=0.05)

# Iterate over each pair of tags to plot them in the same row
for i, (actual_tag, predicted_tag) in enumerate(tags_pairs, start=1):
    for j, tag in enumerate([actual_tag, predicted_tag], start=1):
        histograms = ea.Histograms(tag)

        # Keep only the last N_HISTOGRAMS histograms
        if len(histograms) > N_HISTOGRAMS:
            histograms = histograms[-N_HISTOGRAMS:]

        for histogram_index, histogram in enumerate(histograms):
            step = histogram.step
            bin_edges = histogram.histogram_value.bucket_limit
            counts = histogram.histogram_value.bucket

            # Assign color based on histogram index to ensure same step has same color
            color = colors[histogram_index % N_HISTOGRAMS]

            # Show legend only for the first tag to avoid duplicates
            showlegend = (i == 1 and j == 1)

            # Adding trace for each histogram, with specified bar width
            fig.add_trace(
                go.Bar(
                    x=bin_edges,
                    y=counts,
                    name=f"Step {histogram_index}",
                    width=0.01,
                    marker_color=color,
                    legendgroup=f"Step {histogram_index}",
                    showlegend=showlegend
                ),
                row=i,
                col=j
            )

# Update layout
fig.update_layout(height=1000 if not SIMPLIFIED else 500, title_text="Histograms actual vs predicted values", showlegend=True)

# Update x-axis properties for all plots
fig.update_xaxes(range=[0, 1])

# Update traces opacity
fig.update_traces(opacity=0.75)

# Show the figure
fig.show()
