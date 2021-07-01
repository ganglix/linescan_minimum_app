import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
# data input from user
st.sidebar.header("Parameter settings")
# front end main page
header = st.beta_container()
with header:
    st.title("GPR line scan app")
    st.text(
        "This app helps you visualize the GPR line scan data and measure depths from it."
    )


# sidebar parameter setting
velocity = st.sidebar.number_input(
    "Velocity [ft/ns], default dry soil", value=0.394, step=0.001, format="%.3f"
)
c_vel = 0.3 * 3.28084  # ft/ns


# Read data
uploaded_file = st.file_uploader(
    "Single File Uploader", accept_multiple_files=False, type=".csv"
)


class ScanData:
    pass


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        signal = df.values[2:, 2:].astype(float)
        pos = df.values[0, 2:].astype(float)  # ft
        time = df.values[2:, 1].astype(float)  # ns
    except:
        st.text("file format error, please refer to the demo data format")
        st.stop()
    show_full_data = st.checkbox("check full data file", value=False)
    if show_full_data:
        st.write(df)
    else:
        st.text("preview first five lines")
        st.write(df.head(5))

    if uploaded_file.name.split(".")[0][-1] == "r":
        pos = pos[::-1]
        signal = np.fliplr(signal)
else:
    st.text("To start, please upload a file or use the demo data")
    if st.checkbox("use the demo data", False):
        f = "linex1.csv"
        df = pd.read_csv("./data/" + f)
        signal = df.values[2:, 2:].astype(float)
        pos = df.values[0, 2:].astype(float)  # ft
        time = df.values[2:, 1].astype(float)  # ns
        show_full_data = st.checkbox("check full data file", value=False)
        if show_full_data:
            st.write(df)
        else:
            st.text("preview first five lines")
            st.write(df.head(5))

        if f.split(".")[0][-1] == "r":
            pos = pos[::-1]
            signal = np.fliplr(signal)
    else:
        st.stop()


# process data
La = 0.5 * time * c_vel
L = 0.5 * time * velocity
# kappa = (La/L)**2
# kappa = (c/v)**2

# wrap data
scan = ScanData()
scan.pos = pos
scan.time = time
scan.signal = signal
scan.L = L
scan.La = La


# helper functions
# plot function
# @st.cache(allow_output_mutation=True)
def plot_line(scan):
    # line visualization from interpolated line scan data
    # control parameters

    # plot slice through depth
    fig, ax1 = plt.subplots(figsize=(5, 3))
    # plot line scan
    cf1 = ax1.contourf(
        *np.meshgrid(scan.pos, scan.L),
        scan.signal,
        levels=np.linspace(scan.signal.min(), scan.signal.max(), 100),
        cmap="gray",
    )

    ax1.invert_yaxis()
    ax1.set_xlabel("Position(X or Y) [ft]")
    ax1.set_ylabel("depth [ft]")
    cb1 = plt.colorbar(cf1, ax=ax1)
    ax1.set_title(f"line")
    plt.tight_layout()
    return fig


# Present plot
visualization = st.beta_container()
control = st.beta_container()

with visualization:
    st.header("Visualization")

# measuring lines
# show depth range

with control:
    L_max = scan.L[-1]
    L_min = scan.L[0]
    add_line1 = st.checkbox("add line1", True)
    line1_depth = st.slider(
        "line 1 depth [ft]", L_min, L_max, 0., 0.001, format="%.3f"
    )

    add_line2 = st.checkbox("add line2", True)
    line2_depth = st.slider(
        "line 2 depth [ft]", L_min, L_max, 0., 0.001, format="%.3f"
    )

with visualization:
    st.subheader("line plot")

    fig1 = plot_line(scan)
    if add_line1:
        fig1.axes[0].hlines(line1_depth, 0, scan.pos[-1], "r", lw=0.5)

    if add_line2:
        fig1.axes[0].hlines(line2_depth, 0, scan.pos[-1], "b", lw=0.5)
    st.pyplot(fig1)

# show results in numbers
col1, col2 = st.beta_columns(2)
with col1:
    col1.subheader("line position")
    st.text(f"line 1(Red) depth: {line1_depth:.3f} ft")
    st.text(f"line 2(Blue) depth: {line2_depth:.3f} ft")
    st.text(
        f"line1,2 spacing: {line2_depth - line1_depth:.3f} ft"
    )
with col2:
    col2.subheader("unit conversion ft -> mm")
    num_ft = st.number_input(
        "ft",
        value=line2_depth - line1_depth,
        step=0.001,
        format="%.3f",
    )
    st.text(f"{num_ft*304.8 :.1f} mm ")
