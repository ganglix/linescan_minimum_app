import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")
# data input from user
st.sidebar.header("Parameter settings")

velocity = st.sidebar.number_input("Velocity [ft/ns], default dry soil",value=0.394, step=0.001,format="%.3f")
c_vel = 0.3 * 3.28084 # ft/ns

##############
# prepare data
uploaded_file = st.file_uploader("Single File Uploader", accept_multiple_files=False, type='.csv')

class ScanData: pass

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    signal = df.values[2:,2:].astype(float)
    pos = df.values[0,2:].astype(float) #ft
    time = df.values[2:,1].astype(float) # ns
    show_full_data = st.checkbox("check full data file", value=False)
    if show_full_data:
        st.write(df)
    else:
        st.text("preview first five lines")
        st.write(df.head(5))


    if uploaded_file.name.split('.')[0][-1] == 'r':
        pos = pos[::-1]
        signal = np.fliplr(signal)
        
    # derived
    La = 0.5* time * c_vel
    L = 0.5* time* velocity
    # kappa = (La/L)**2
    # kappa = (c/v)**2
        
    # wrap data
    scan = ScanData()
    scan.pos = pos
    scan.time = time
    scan.signal = signal
    scan.L = L
    scan.La = La
else:
    st.text('file error')
    st.stop()

# @st.cache(allow_output_mutation=True) 
def plot_line(scan):
    # line visualization from interpolated line scan data
    # control parameters

    # plot slice through depth
    fig, ax1 = plt.subplots(figsize=(6,4))
    # plot line scan
    cf1 = ax1.contourf(*np.meshgrid(scan.pos, scan.L),
                     scan.signal, 
                     levels=np.linspace(scan.signal.min(), scan.signal.max(),100), 
                     cmap='gray')

    ax1.invert_yaxis()
    ax1.set_xlabel('deck width, X [ft]')
    ax1.set_ylabel('depth [ft]')
    cb1 = plt.colorbar(cf1, ax=ax1)
    ax1.set_title(f'line')
    plt.tight_layout()
    return fig

# front end main page
header = st.beta_container()

visualization = st.beta_container()



with header:
    st.title('GPR app')
    st.text('This app helps you to visualize the GPR scan data')
    
with visualization:
    st.header('Visualization')


# measuring lines
# show depth range

add_1st_line = st.checkbox("add_line1", True)
depth_point_1st = st.sidebar.slider('line 1 depth point', 0, len(scan.time),0,1)

add_2nd_line = st.checkbox("add_line2", True)
depth_point_2nd = st.sidebar.slider('line 2 depth point', 0, len(scan.time),0,1)

with visualization:
    st.subheader('line plot')

    fig1 = plot_line(scan)
    if add_1st_line:
        fig1.axes[0].hlines(L[depth_point_1st],0,scan.pos[-1],'r', lw=0.5)

    if add_2nd_line:
        fig1.axes[0].hlines(L[depth_point_2nd],0,scan.pos[-1],'b', lw=0.5)
    st.pyplot(fig1)

col1,col2 = st.beta_columns(2)
with col1:
    col1.subheader("line position")
    st.text(f"line 1(Red) depth: {scan.L[depth_point_1st]:.3f} ft")
    st.text(f"line 2(Blue) depth: {scan.L[depth_point_2nd]:.3f} ft")
    st.text(f"line1,2 spacing: {scan.L[depth_point_2nd]-scan.L[depth_point_1st]:.3f} ft")
with col2:
    col2.subheader('unit conversion ft -> mm')
    num_ft = st.number_input("ft", step=0.001,format="%.3f")
    st.text(f"{num_ft*304.8 :.1f} mm ")
