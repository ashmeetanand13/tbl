import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import tempfile
import os
import re

# Configure page and initialize session state
st.set_page_config(page_title="Chamber Pressure Analysis", layout="wide")

# Initialize session state for progressive loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'combined_df' not in st.session_state:
    st.session_state.combined_df = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'chamber_mappings' not in st.session_state:
    st.session_state.chamber_mappings = {}
if 'selected_chambers' not in st.session_state:
    st.session_state.selected_chambers = []

# App title
st.title("Chamber Pressure Analysis Tool")
st.write("Upload your chamber data files to analyze pressure cycles.")

# Function to extract chamber number from filename - with caching
@st.cache_data
def extract_chamber_number(filename):
    """
    Extract chamber number from filename (e.g., C1.xls -> 1, Chamber2.csv -> 2)
    """
    pattern = r'[Cc](?:hamber)?(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        return int(match.group(1))
    else:
        return None

# Optimized file reading function - with caching
@st.cache_data
def read_file(file_content, file_name):
    """
    Efficiently read file content
    """
    # Create a BytesIO object from the file content
    file_io = io.BytesIO(file_content)
    
    try:
        # Try Excel first for .xls/.xlsx
        if file_name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_io)
            
        # Try CSV with comma delimiter
        try:
            df = pd.read_csv(file_io)
            if len(df.columns) > 1:  # Success if multiple columns
                return df
            # Reset position for next attempt
            file_io.seek(0)
        except:
            pass
        
        # Try tab delimiter
        try:
            df = pd.read_csv(file_io, sep='\t')
            if len(df.columns) > 1:
                return df
            file_io.seek(0)
        except:
            pass
            
        # Try semicolon delimiter (common in European data)
        try:
            df = pd.read_csv(file_io, sep=';')
            if len(df.columns) > 1:
                return df
            file_io.seek(0)
        except:
            pass
        
        # Last resort - let pandas detect
        return pd.read_csv(file_io, engine='python')
            
    except Exception as e:
        raise Exception(f"Failed to read {file_name}: {str(e)}")

# Optimized cycle analysis function - with caching
@st.cache_data
def analyze_chamber_cycles(df, chamber_col, start_threshold, end_threshold):
    """
    Analyze cycles for a specific chamber using vectorized operations
    """
    # Make a copy to avoid modifying the original
    analysis_df = df[['Time', chamber_col]].copy().dropna()
    
    # Skip if empty
    if analysis_df.empty:
        return []
    
    # Ensure sorted by time
    analysis_df = analysis_df.sort_values('Time').reset_index(drop=True)
    
    # Find points where values cross thresholds
    above_start = analysis_df[chamber_col] > start_threshold
    below_end = analysis_df[chamber_col] < end_threshold
    
    # Detect state changes (False to True)
    start_cycle = above_start & ~above_start.shift(1, fill_value=False)
    end_cycle = below_end & ~below_end.shift(1, fill_value=False)
    
    # Get indices where cycles start
    cycle_starts = analysis_df.index[start_cycle].tolist()
    
    # List to store cycle info
    cycles = []
    
    # Process each detected cycle
    for start_idx in cycle_starts:
        # Find the next cycle end after this start
        end_indices = analysis_df.index[end_cycle & (analysis_df.index > start_idx)].tolist()
        if not end_indices:
            # No end found, use the last data point
            end_idx = analysis_df.index[-1]
        else:
            end_idx = end_indices[0]
            
        # Get cycle segment
        cycle_segment = analysis_df.loc[start_idx:end_idx]
        
        # Find the peak (maximum value)
        max_idx = cycle_segment[chamber_col].idxmax()
        max_value = cycle_segment.loc[max_idx, chamber_col]
        max_time = cycle_segment.loc[max_idx, 'Time']
        
        # Get start and end times
        start_time = analysis_df.loc[start_idx, 'Time']
        end_time = analysis_df.loc[end_idx, 'Time']
        
        # Calculate duration
        try:
            duration = (end_time - start_time).total_seconds()
        except:
            duration = np.nan
            
        # Store cycle
        cycle = {
            'max_value': max_value,
            'max_time': max_time,
            'start_time': start_time,
            'end_time': end_time,
            'peak_time': max_time,
            'duration': duration
        }
        
        cycles.append(cycle)
    
    return cycles

# Function to prepare dataframe - with caching
@st.cache_data
def prepare_dataframe(df):
    """
    Efficiently prepare and clean dataframe
    """
    # Handle duplicate columns
    if len(df.columns) != len(set(df.columns)):
        # Make a copy with unique column names
        new_cols = []
        seen = set()
        for i, col in enumerate(df.columns):
            if col in seen:
                new_cols.append(f"{col}_{i}")
            else:
                new_cols.append(col)
                seen.add(col)
        df.columns = new_cols
        
        # Fix 'Time' column specifically
        time_cols = [col for col in new_cols if col.startswith('Time')]
        if time_cols and 'Time' not in time_cols:
            # Rename first Time_x back to Time
            df.rename(columns={time_cols[0]: 'Time'}, inplace=True)
    
    # Convert Time to datetime if not already
    if 'Time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        
    return df

# Function to combine dataframes - with caching
@st.cache_data
def combine_dataframes(dataframes):
    """
    Efficiently combine dataframes on Time column
    """
    if not dataframes:
        return pd.DataFrame()
        
    # Start with first dataframe
    combined_df = dataframes[0].copy()
    
    # Merge with each additional dataframe
    for i in range(1, len(dataframes)):
        combined_df = pd.merge(
            combined_df,
            dataframes[i],
            on='Time',
            how='outer'
        )
    
    # Sort and reset index
    combined_df = combined_df.sort_values('Time').reset_index(drop=True)
    return combined_df

# Create a placeholder for status messages
status = st.empty()

# Sidebar for file uploads and parameters
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose Excel/CSV files", accept_multiple_files=True, type=['xls', 'xlsx', 'csv'])
    
    st.header("Cycle Parameters")
    st.write("Set thresholds for chamber pressure cycles")
    start_threshold = st.number_input("Start Threshold", value=3.0, step=0.1, 
                                      help="Pressure above which a cycle begins")
    end_threshold = st.number_input("End Threshold", value=3.0, step=0.1,
                                    help="Pressure below which a cycle ends")
    
    # Simplified advanced options
    show_debug_info = st.checkbox("Show Debug Information", value=False,
                                 help="Enable to see detailed information for troubleshooting")

# Main content area
if uploaded_files:
    if not st.session_state.data_loaded:
        # Process files only if not already loaded
        with st.spinner("Processing files..."):
            status.text("Reading files...")
            
            # Set up progress bar
            progress = st.progress(0)
            
            # Process files
            dataframes = []
            chamber_mappings = {}
            
            # Create a processing container to hide details later
            processing_container = st.container()
            
            with processing_container:
                if show_debug_info:
                    st.subheader("File Processing")
                
                # Process each file
                for i, file in enumerate(uploaded_files):
                    try:
                        # Extract chamber number
                        chamber_num = extract_chamber_number(file.name)
                        chamber_name = f"Chamber {chamber_num}" if chamber_num else f"Unknown Chamber {i+1}"
                        
                        if show_debug_info:
                            st.write(f"📊 Processing: {file.name} - {chamber_name}")
                        
                        # Read the file directly from memory
                        file_content = file.read()
                        df = read_file(file_content, file.name)
                        
                        # Debug output
                        if show_debug_info:
                            st.write(f"Columns found: {', '.join(df.columns.tolist())}")
                        
                        # Identify potential time and value columns
                        time_col = None
                        value_col = None
                        
                        # Find time column
                        if 'Time' in df.columns:
                            time_col = 'Time'
                        else:
                            # Look for columns that might contain time
                            time_candidates = [col for col in df.columns if 
                                             any(kw in str(col).lower() for kw in ['time', 'date', 'timestamp'])]
                            if time_candidates:
                                time_col = time_candidates[0]
                        
                        # If still no time column, let user select
                        if not time_col:
                            # Find columns with datetime or numeric types as candidates
                            candidates = []
                            for col in df.columns:
                                if df[col].dtype in ['datetime64[ns]', 'int64', 'float64', 'object']:
                                    candidates.append(col)
                            
                            if candidates:
                                time_col = st.selectbox(
                                    f"Select time column for {file.name}",
                                    options=candidates,
                                    key=f"time_col_{i}"
                                )
                            else:
                                time_col = st.selectbox(
                                    f"Select time column for {file.name}",
                                    options=df.columns.tolist(),
                                    key=f"time_col_{i}"
                                )
                        
                        # Find value column
                        if 'Value' in df.columns:
                            value_col = 'Value'
                        else:
                            # Look for columns that might contain values
                            value_candidates = [col for col in df.columns if 
                                              any(kw in str(col).lower() for kw in ['value', 'pressure'])]
                            if value_candidates:
                                value_col = value_candidates[0]
                            else:
                                # Get numeric columns as candidates
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                numeric_cols = [col for col in numeric_cols if col != time_col and not col.startswith('Unnamed')]
                                
                                if numeric_cols:
                                    value_col = st.selectbox(
                                        f"Select value column for {file.name}",
                                        options=numeric_cols,
                                        key=f"value_col_{i}"
                                    )
                                else:
                                    st.error(f"No numeric columns found in {file.name}")
                                    continue
                        
                        # Rename time column
                        if time_col and time_col != 'Time':
                            df.rename(columns={time_col: 'Time'}, inplace=True)
                        
                        # Create simplified dataframe with just Time and value
                        if value_col:
                            # Rename value column based on chamber
                            new_col_name = chamber_name
                            
                            # Store mapping
                            chamber_mappings[new_col_name] = file.name
                            
                            # Create simplified df
                            simple_df = df[['Time', value_col]].copy()
                            simple_df.rename(columns={value_col: new_col_name}, inplace=True)
                            
                            # Add to dataframes list
                            dataframes.append(simple_df)
                            
                            if show_debug_info:
                                sample = df[value_col].dropna().head(3).tolist()
                                st.write(f"Sample values: {sample}")
                        
                        # Update progress
                        progress.progress((i + 1) / len(uploaded_files))
                        status.text(f"Processed {i+1}/{len(uploaded_files)} files")
                        
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
                
                # Save chamber mappings to session state
                st.session_state.chamber_mappings = chamber_mappings
            
            # Combine dataframes and prepare
            if dataframes:
                status.text("Combining data...")
                combined_df = combine_dataframes(dataframes)
                combined_df = prepare_dataframe(combined_df)
                
                # Store in session state
                st.session_state.combined_df = combined_df
                st.session_state.data_loaded = True
                status.text("Data loaded successfully!")
            else:
                st.error("No data could be processed from the uploaded files.")

    # If data is loaded, display analysis options
    if st.session_state.data_loaded and st.session_state.combined_df is not None:
        combined_df = st.session_state.combined_df
        chamber_mappings = st.session_state.chamber_mappings
        
        # Show chamber mappings
        st.subheader("Chamber Mappings")
        for chamber, filename in chamber_mappings.items():
            st.write(f"📊 **{chamber}** - Data from file: {filename}")
        
        # Display preview
        st.subheader("Data Preview")
        st.write(combined_df.head())
        
        # Get all chamber columns
        chamber_cols = [col for col in combined_df.columns if col != 'Time']
        
        if chamber_cols:
            # Select chambers
            st.subheader("Chamber Selection")
            selected_chambers = st.multiselect(
                "Select chambers to analyze",
                chamber_cols,
                default=chamber_cols
            )
            st.session_state.selected_chambers = selected_chambers
            
            # Run analysis button
            if selected_chambers and st.button("Analyze Selected Chambers"):
                with st.spinner("Analyzing chamber cycles..."):
                    status.text("Analyzing chamber cycles...")
                    
                    # Analyze each chamber
                    results = {}
                    for chamber in selected_chambers:
                        if chamber in combined_df.columns:
                            results[chamber] = analyze_chamber_cycles(combined_df, chamber, start_threshold, end_threshold)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                    status.text("Analysis complete!")
            
            # Display results if analysis complete
            if st.session_state.analysis_complete and st.session_state.results:
                results = st.session_state.results
                
                # Extract cycle data
                cycle_data = []
                for chamber, cycles in results.items():
                    for i, cycle in enumerate(cycles, 1):
                        cycle_entry = {
                            'Chamber': chamber,
                            'Cycle Number': i,
                            'Max Value': cycle['max_value'],
                            'Max Time': cycle['max_time'],
                            'Cycle Duration (seconds)': cycle['duration'],
                            'Start Time': cycle['start_time'],
                            'End Time': cycle['end_time'],
                            'Peak Time': cycle['peak_time']
                        }
                        cycle_data.append(cycle_entry)
                
                # Create DataFrame
                if cycle_data:
                    df_cycles = pd.DataFrame(cycle_data)
                    
                    # Display results
                    st.subheader("Chamber Cycle Analysis Results")
                    st.write(df_cycles)
                    
                    # Display summary
                    st.subheader("Cycles per Chamber")
                    cycles_summary = df_cycles.groupby('Chamber').size().reset_index(name='Number of Cycles')
                    st.write(cycles_summary)
                    
                    # Time range for visualization
                    min_time = combined_df['Time'].min()
                    max_time = combined_df['Time'].max()
                    
                    if not pd.isna(min_time) and not pd.isna(max_time):
                        st.subheader("Visualization Options")
                        
                        # Date selection
                        col1, col2 = st.columns(2)
                        with col1:
                            start_date = st.date_input("Start date", min_time.date())
                        with col2:
                            end_date = st.date_input("End date", max_time.date())
                        
                        # Convert to timestamps
                        start_time = pd.Timestamp(start_date)
                        end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        
                        # Create visualization
                        if st.button("Generate Visualizations"):
                            with st.spinner("Creating visualizations..."):
                                # Filter data
                                filtered_df = combined_df[(combined_df['Time'] >= start_time) & 
                                                         (combined_df['Time'] <= end_time)]
                                
                                # Create color map
                                colors = px.colors.qualitative.Plotly[:len(selected_chambers)]
                                color_map = {chamber: color for chamber, color in zip(selected_chambers, colors)}
                                
                                # Main visualization
                                st.subheader("Chamber Pressure Visualization")
                                
                                # Create figure
                                fig = go.Figure()
                                
                                # Add traces
                                for chamber in selected_chambers:
                                    if chamber in filtered_df.columns:
                                        fig.add_trace(go.Scatter(
                                            x=filtered_df['Time'],
                                            y=filtered_df[chamber],
                                            mode='lines',
                                            name=chamber,
                                            line=dict(
                                                color=color_map.get(chamber, 'blue'),
                                                width=2
                                            )
                                        ))
                                
                                # Add threshold lines
                                fig.add_shape(
                                    type="line",
                                    x0=start_time,
                                    y0=start_threshold,
                                    x1=end_time,
                                    y1=start_threshold,
                                    line=dict(color="green", width=2, dash="dash")
                                )
                                
                                fig.add_shape(
                                    type="line",
                                    x0=start_time,
                                    y0=end_threshold,
                                    x1=end_time,
                                    y1=end_threshold,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                # Add annotations
                                fig.add_annotation(
                                    x=start_time,
                                    y=start_threshold,
                                    text=f"Start: {start_threshold}",
                                    showarrow=False,
                                    yshift=10,
                                    bgcolor="rgba(255,255,255,0.8)"
                                )
                                
                                fig.add_annotation(
                                    x=start_time,
                                    y=end_threshold,
                                    text=f"End: {end_threshold}",
                                    showarrow=False,
                                    yshift=-20,
                                    bgcolor="rgba(255,255,255,0.8)"
                                )
                                
                                # Update layout
                                fig.update_layout(
                                    title="Chamber Pressure Over Time",
                                    xaxis_title="Time",
                                    yaxis_title="Pressure (mmHg)",
                                    height=500,
                                    hovermode="x unified",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Show plot
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Plot with cycle markers
                                if not df_cycles.empty:
                                    st.subheader("Chamber Pressure Cycles")
                                    
                                    # Create figure
                                    cycle_fig = go.Figure()
                                    
                                    # Add traces
                                    for chamber in selected_chambers:
                                        if chamber in filtered_df.columns:
                                            # Main line
                                            cycle_fig.add_trace(go.Scatter(
                                                x=filtered_df['Time'],
                                                y=filtered_df[chamber],
                                                mode='lines',
                                                name=chamber,
                                                line=dict(
                                                    color=color_map.get(chamber, 'blue'),
                                                    width=2
                                                )
                                            ))
                                            
                                            # Cycle markers
                                            chamber_cycles = df_cycles[df_cycles['Chamber'] == chamber]
                                            if not chamber_cycles.empty:
                                                # Filter to time range
                                                chamber_cycles = chamber_cycles[
                                                    (chamber_cycles['Max Time'] >= start_time) & 
                                                    (chamber_cycles['Max Time'] <= end_time)
                                                ]
                                                
                                                if not chamber_cycles.empty:
                                                    cycle_fig.add_trace(go.Scatter(
                                                        x=chamber_cycles['Max Time'],
                                                        y=chamber_cycles['Max Value'],
                                                        mode='markers',
                                                        marker=dict(
                                                            color=color_map.get(chamber, 'blue'),
                                                            size=10,
                                                            symbol='star'
                                                        ),
                                                        name=f"{chamber} Peaks",
                                                        text=[f"Cycle {i}" for i in chamber_cycles['Cycle Number']],
                                                        hoverinfo='text+y'
                                                    ))
                                    
                                    # Add thresholds
                                    cycle_fig.add_shape(
                                        type="line",
                                        x0=start_time,
                                        y0=start_threshold,
                                        x1=end_time,
                                        y1=start_threshold,
                                        line=dict(color="green", width=2, dash="dash")
                                    )
                                    
                                    cycle_fig.add_shape(
                                        type="line",
                                        x0=start_time,
                                        y0=end_threshold,
                                        x1=end_time,
                                        y1=end_threshold,
                                        line=dict(color="red", width=2, dash="dash")
                                    )
                                    
                                    # Update layout
                                    cycle_fig.update_layout(
                                        title="Chamber Pressure Cycles with Peak Markers",
                                        xaxis_title="Time",
                                        yaxis_title="Pressure (mmHg)",
                                        height=500,
                                        hovermode="closest",
                                        legend=dict(
                                            orientation="h",
                                            yanchor="bottom",
                                            y=1.02,
                                            xanchor="right",
                                            x=1
                                        )
                                    )
                                    
                                    # Show plot
                                    st.plotly_chart(cycle_fig, use_container_width=True)
                    
                    # Export options
                    st.subheader("Export Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export combined data
                        st.download_button(
                            label="Download Combined Data (CSV)",
                            data=combined_df.to_csv(index=False).encode('utf-8'),
                            file_name="combined_data.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Export cycle analysis
                        st.download_button(
                            label="Download Cycle Analysis (CSV)",
                            data=df_cycles.to_csv(index=False).encode('utf-8'),
                            file_name="cycle_analysis.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.warning(f"No cycles detected with current threshold settings. Try adjusting thresholds.")

        # Add reset button
        if st.button("Reset Analysis"):
            # Clear session state
            st.session_state.data_loaded = False
            st.session_state.analysis_complete = False
            st.session_state.combined_df = None
            st.session_state.results = None
            st.session_state.chamber_mappings = {}
            st.session_state.selected_chambers = []
            st.experimental_rerun()
            
else:
    st.info("Please upload files to begin analysis")
    st.write("""
    ### Instructions:
    1. Upload your chamber data files (Excel or CSV format)
    2. Set the start and end threshold values for cycle detection
    3. Select chambers to analyze
    4. Click "Analyze Selected Chambers"
    5. Generate visualizations and export results if needed
    
    ### File Naming:
    - Name your files to indicate which chamber they represent (e.g., C1.xls, Chamber2.csv)
    - The app will automatically extract the chamber number from the filename
    
    ### Performance Tips:
    - This optimized version uses caching to significantly improve loading times
    - Use the "Reset Analysis" button if you need to start over with new files
    - Generate visualizations only when needed to improve performance
    """)
