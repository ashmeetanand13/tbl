import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
import tempfile
import os
import re

st.set_page_config(page_title="Chamber Pressure Analysis", layout="wide")

# App title
st.title("Chamber Pressure Analysis Tool")
st.write("Upload your chamber data files to analyze pressure cycles.")


# Function to extract chamber number from filename
def extract_chamber_number(filename):
    """
    Extract chamber number from filename (e.g., C1.xls -> 1, Chamber2.csv -> 2)
    """
    # Try to find patterns like C1, C2, Chamber1, Chamber2, etc.
    pattern = r'[Cc](?:hamber)?(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        return int(match.group(1))
    else:
        # If no pattern found, return None and we'll handle this case later
        return None


# Function to read a file using multiple methods
def read_file(file_path, file_name):
    """
    Attempt to read a file using multiple methods
    """
    # First try: Read as Excel
    if file_name.endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            st.info(f"Could not read {file_name} as Excel. Trying alternative formats...")
            # Fall through to next method
    
    # Second try: Read as CSV with comma delimiter
    try:
        df = pd.read_csv(file_path, sep=',')
        # Check if we got a single column (might be wrong delimiter)
        if len(df.columns) == 1:
            raise Exception("Only one column detected with comma delimiter, trying tab delimiter")
        return df
    except Exception as e:
        # Third try: Read as CSV with tab delimiter
        try:
            df = pd.read_csv(file_path, sep='\t')
            return df
        except Exception as e:
            # Fourth try: Try to detect delimiter
            try:
                with open(file_path, 'r') as f:
                    sample = f.read(1024)
                if '\t' in sample:
                    return pd.read_csv(file_path, sep='\t')
                elif ';' in sample:
                    return pd.read_csv(file_path, sep=';')
                elif ',' in sample:
                    return pd.read_csv(file_path, sep=',')
                else:
                    return pd.read_csv(file_path, engine='python')  # Let pandas detect
            except Exception as e:
                raise Exception(f"Failed to read {file_name}: {str(e)}")


# Function to analyze chamber cycles
def analyze_chamber_cycles(df, chamber_col, start_threshold, end_threshold):
    """
    Analyze cycles for a specific chamber
    - start_threshold: pressure above which a cycle begins
    - end_threshold: pressure below which a cycle ends
    """
    cycles = []
    current_cycle = {
        'max_value': float('-inf'), 
        'max_time': None, 
        'start_time': None, 
        'end_time': None,
        'peak_time': None
    }
    
    in_cycle = False
    
    # Make sure chamber_col exists in the DataFrame
    if chamber_col not in df.columns:
        st.error(f"Column '{chamber_col}' not found in the data.")
        return []
    
    # Convert DataFrame to use .iterrows() instead of direct indexing
    # This helps avoid the "truth value of a Series is ambiguous" error
    for idx, row in df.iterrows():
        try:
            value = row[chamber_col]
            time = row['Time']
            
            # Skip rows with NaN time or value
            if pd.isna(time) or pd.isna(value):
                continue
            
            # Start a new cycle when pressure rises above start_threshold
            if not in_cycle and value > start_threshold:
                in_cycle = True
                current_cycle['start_time'] = time
                current_cycle['max_value'] = value
                current_cycle['max_time'] = time
                current_cycle['peak_time'] = time
            
            # While in a cycle, track the maximum value
            if in_cycle:
                if value > current_cycle['max_value']:
                    current_cycle['max_value'] = value
                    current_cycle['max_time'] = time
                    current_cycle['peak_time'] = time
            
            # End the cycle when pressure drops below end_threshold
            if in_cycle and value < end_threshold:
                current_cycle['end_time'] = time
                
                # Calculate duration if both start and end times are valid
                if isinstance(current_cycle['start_time'], pd.Timestamp) and isinstance(current_cycle['end_time'], pd.Timestamp):
                    current_cycle['duration'] = (current_cycle['end_time'] - current_cycle['start_time']).total_seconds()
                else:
                    # Handle case where times might not be proper timestamp objects
                    try:
                        # Try to convert to timestamp if they're strings
                        start = pd.to_datetime(current_cycle['start_time'])
                        end = pd.to_datetime(current_cycle['end_time'])
                        current_cycle['duration'] = (end - start).total_seconds()
                    except:
                        current_cycle['duration'] = np.nan
                
                # Store the completed cycle
                cycles.append(current_cycle.copy())
                
                # Reset for next cycle
                in_cycle = False
                current_cycle = {
                    'max_value': float('-inf'), 
                    'max_time': None, 
                    'start_time': None, 
                    'end_time': None,
                    'peak_time': None
                }
        except Exception as e:
            st.warning(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Handle case where data ends while still in a cycle
    if in_cycle and current_cycle['start_time'] is not None:
        # Get the last valid time value instead of using loc indexing
        last_valid_time = df['Time'].dropna().iloc[-1] if not df['Time'].dropna().empty else None
        
        current_cycle['end_time'] = last_valid_time
        
        if current_cycle['end_time'] is not None:
            try:
                start = pd.to_datetime(current_cycle['start_time'])
                end = pd.to_datetime(current_cycle['end_time'])
                current_cycle['duration'] = (end - start).total_seconds()
            except:
                current_cycle['duration'] = np.nan
            cycles.append(current_cycle.copy())
    
    return cycles


# Prepare dataframe - Convert Time column to datetime
def prepare_dataframe(df):
    # Check for duplicate column names and fix them
    if len(df.columns) != len(set(df.columns)):
        # Get all duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated(keep=False)]
        
        # Handle duplicate 'Time' columns specifically
        time_cols = [col for col in duplicate_cols if col == 'Time']
        
        if time_cols:
            st.warning(f"Found duplicate 'Time' columns: {len(time_cols)}. Keeping only the first one.")
            
            # Create a new DataFrame with unique column names
            new_cols = []
            seen = set()
            for i, col in enumerate(df.columns):
                if col in seen:
                    # For duplicates, add a suffix
                    new_cols.append(f"{col}_{i}")
                else:
                    new_cols.append(col)
                    seen.add(col)
            
            # Set new column names
            df.columns = new_cols
            
            # Identify the original 'Time' column (should be the first one)
            time_col = 'Time'
            if time_col not in df.columns:
                # If we renamed the first Time column, find it
                time_col = [col for col in new_cols if col.startswith('Time_')][0]
            
            # Rename back to 'Time' if needed
            if time_col != 'Time':
                df.rename(columns={time_col: 'Time'}, inplace=True)
    
    # Now ensure we have a 'Time' column
    if 'Time' not in df.columns:
        st.error("No 'Time' column found after processing. Please check your input files.")
        return df
    
    # Convert Time column to datetime, handling errors
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    return df


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
    
    # Note: Only merge on time method is used in this version
    st.write("**Files will be merged on matching timestamps**")
    
    # Advanced options
    st.header("Advanced Options")
    show_debug_info = st.checkbox("Show Debug Information", value=False,
                                 help="Enable to see detailed information for troubleshooting")
    
    if show_debug_info:
        st.info("Debug mode enabled. Additional diagnostic information will be shown.")
        st.header("File Processing Options")
        skip_time_conversion = st.checkbox("Skip Time Conversion", value=False,
                                         help="Enable if time conversion is causing problems")
        custom_time_column = st.text_input("Force Time Column Name", value="",
                                         help="Enter a specific column name to use as time")
        custom_value_prefix = st.text_input("Value Column Prefix", value="",
                                          help="Enter prefix for identifying value columns (e.g., 'Value')")
        
        # Export options
        dump_data = st.checkbox("Export Raw Data", value=False,
                             help="Export data as seen by the app for debugging")
        if dump_data:
            st.download_button(label="Download Debug Info", 
                               data="Debug information will be generated when files are processed.",
                               file_name="debug_info.txt",
                               disabled=True)


# Main content
if uploaded_files:
    st.header("Data Processing")
    status = st.empty()
    
    # Dictionary to store data processing progress
    progress = st.progress(0)
    
    # Get debug options
    use_custom_time = False
    if show_debug_info and custom_time_column:
        use_custom_time = True
        st.info(f"Using custom time column: {custom_time_column}")
    
    # Get custom value prefix if provided
    value_prefix = ""
    if show_debug_info and custom_value_prefix:
        value_prefix = custom_value_prefix
        st.info(f"Using custom value column prefix: {value_prefix}")
    
    with st.spinner("Processing files..."):
        status.text("Reading files...")
        
        # Process files
        dataframes = []
        chamber_mappings = {}  # Map to store which file corresponds to which chamber
        
        # Debug information
        if show_debug_info:
            st.subheader("File Processing Debug Information")
        
        # Process each uploaded file
        for i, file in enumerate(uploaded_files):
            # Create a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            # Extract chamber number from filename
            chamber_num = extract_chamber_number(file.name)
            if chamber_num:
                chamber_name = f"Chamber {chamber_num}"
                st.write(f"📊 Processing file: {file.name} - Identified as {chamber_name}")
            else:
                chamber_name = f"Unknown Chamber {i+1}"
                st.write(f"📊 Processing file: {file.name} - Could not identify chamber number, using {chamber_name}")
            
            # If in debug mode, show file info
            if show_debug_info:
                # Try to read first few lines of the file
                try:
                    with open(tmp_path, 'r') as f:
                        first_lines = [next(f) for _ in range(5)]
                        st.code('\n'.join(first_lines), language="text")
                except:
                    st.warning("Could not read file as text. It may be a binary format.")
            
            try:
                # Read file using our custom function
                df = read_file(tmp_path, file.name)
                
                # Display file information for debugging
                st.write(f"### File: {file.name}")
                st.write(f"**Columns found:** {', '.join(df.columns.tolist())}")
                
                # Identify possible time columns for user selection
                possible_time_cols = []
                for col in df.columns:
                    # Check for common time column names
                    if any(keyword in str(col).lower() for keyword in ['time', 'date', 'timestamp']):
                        possible_time_cols.append(col)
                    # Check for datetime-like or numeric columns
                    elif df[col].dtype in ['datetime64[ns]', 'int64', 'float64']:
                        # Sample the first few values
                        sample = df[col].dropna().head(3).tolist()
                        st.write(f"**Column '{col}'** - Type: {df[col].dtype}, Sample values: {sample}")
                        possible_time_cols.append(col)
                
                # Let user select the time column
                if use_custom_time and custom_time_column in df.columns:
                    time_col = custom_time_column
                    st.success(f"Using custom time column: {time_col}")
                elif 'Time' in df.columns:
                    time_col = 'Time'  # Default if it exists
                else:
                    if possible_time_cols:
                        time_col = st.selectbox(
                            f"Select time column for {file.name}", 
                            options=possible_time_cols,
                            key=f"time_col_{i}"
                        )
                    else:
                        time_col = st.selectbox(
                            f"Select time column for {file.name}", 
                            options=df.columns.tolist(),
                            key=f"time_col_{i}"
                        )
                
                # Add file identifier column
                df['source_file'] = file.name
                
                # Prepare dataframe (convert time column)
                if time_col != 'Time':
                    df.rename(columns={time_col: 'Time'}, inplace=True)
                
                # Convert Time column to datetime if not skipping time conversion
                if not (show_debug_info and skip_time_conversion) and not pd.api.types.is_datetime64_any_dtype(df['Time']):
                    # Show sample of time values
                    time_sample = df['Time'].dropna().head(5).tolist()
                    st.write(f"**Time column values (before conversion):** {time_sample}")
                    
                    # Let user select time format if needed
                    format_options = [
                        "Auto-detect",
                        "%Y-%m-%d %H:%M:%S",  # 2023-01-30 14:30:45
                        "%m/%d/%y %H:%M",     # 11/26/24 10:25
                        "%d/%m/%Y %H:%M:%S",  # 30/01/2023 14:30:45
                        "%m/%d/%Y %H:%M:%S",  # 01/30/2023 14:30:45
                        "%Y-%m-%d",           # 2023-01-30
                        "%H:%M:%S",           # 14:30:45
                        "%H:%M"               # 14:30
                    ]
                    time_format = st.selectbox(
                        f"Select time format for {file.name}",
                        options=format_options,
                        key=f"time_format_{i}"
                    )
                    
                    try:
                        if time_format == "Auto-detect":
                            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                        else:
                            df['Time'] = pd.to_datetime(df['Time'], format=time_format, errors='coerce')
                        
                        # Show sample after conversion
                        time_converted = df['Time'].dropna().head(5).tolist()
                        st.write(f"**Time column values (after conversion):** {time_converted}")
                        
                    except Exception as e:
                        st.error(f"Error converting time: {str(e)}")
                        st.warning("Will try to use the column as-is")
                elif show_debug_info and skip_time_conversion:
                    st.info("Time conversion skipped as requested")
                
                # Check if we have a valid Time column after conversion
                if df['Time'].isna().all():
                    st.error(f"Time column in {file.name} contains all invalid dates. Check the format.")
                elif df['Time'].isna().any():
                    percent_invalid = (df['Time'].isna().sum() / len(df)) * 100
                    st.warning(f"{percent_invalid:.1f}% of timestamps in {file.name} are invalid.")
                
                # Identify value column (typically named 'Value')
                value_col = None
                if 'Value' in df.columns:
                    value_col = 'Value'
                else:
                    # Try to find a column that might contain values
                    value_cols = [col for col in df.columns if 'value' in col.lower() or 'pressure' in col.lower()]
                    if value_cols:
                        value_col = value_cols[0]  # Use the first one
                        st.write(f"Found potential value column: {value_col}")
                    else:
                        # If no obvious value column, let user select
                        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [col for col in numeric_cols if col != 'Time' and not col.startswith('Unnamed')]
                        if numeric_cols:
                            value_col = st.selectbox(
                                f"Select value column for {file.name}", 
                                options=numeric_cols,
                                key=f"value_col_{i}"
                            )
                            st.write(f"Selected value column: {value_col}")
                        else:
                            st.error(f"No numeric columns found in {file.name} to use as values.")
                            continue  # Skip this file
                
                if value_col:
                    # Rename the value column based on chamber number
                    if chamber_num:
                        new_col_name = f"Chamber {chamber_num}"
                    else:
                        new_col_name = f"Unknown Chamber {i+1}"
                    
                    # Store mapping for later reference
                    chamber_mappings[new_col_name] = file.name
                    
                    # Create a simplified DataFrame with just Time and the value column
                    simple_df = df[['Time', value_col]].copy()
                    simple_df.rename(columns={value_col: new_col_name}, inplace=True)
                    
                    # Add to our list of dataframes
                    dataframes.append(simple_df)
                    
                    # Show sample of value column
                    sample = df[value_col].dropna().head(5).tolist()
                    st.write(f"**Sample values from {value_col} (mapped to {new_col_name}):** {sample}")
                
                # Generate debug dump if requested
                if show_debug_info and dump_data:
                    dump_text = f"File: {file.name}\n"
                    dump_text += f"Chamber: {chamber_name}\n"
                    dump_text += f"Columns: {df.columns.tolist()}\n"
                    dump_text += f"Time Column: {time_col}\n"
                    dump_text += f"Value Column: {value_col}\n"
                    dump_text += f"Data Types: {df.dtypes.to_string()}\n"
                    dump_text += f"First 5 rows:\n{df.head().to_string()}\n\n"
                    
                    st.download_button(
                        label=f"Download Debug Info for {file.name}",
                        data=dump_text,
                        file_name=f"debug_{file.name}.txt"
                    )
                
                # Add separators between files
                st.markdown("---")
                
                # Update progress
                progress.progress((i + 1) / len(uploaded_files))
                status.text(f"Processed {i+1}/{len(uploaded_files)} files")
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            # Clean up the temporary file
            os.unlink(tmp_path)
    
    # Check if any dataframes were successfully loaded
    if not dataframes:
        st.error("No files could be processed. Please check the format of your files.")
    else:
        # Combine the dataframes using merge on time
        status.text("Combining files on matching timestamps...")
        
        # Start with the first dataframe
        combined_df = dataframes[0].copy()
        
        # Merge with each subsequent dataframe on the time column
        for i in range(1, len(dataframes)):
            # Use outer merge to keep all timestamps from both dataframes
            combined_df = pd.merge(
                combined_df, 
                dataframes[i], 
                on='Time', 
                how='outer'
            )
        
        # Sort by Time
        combined_df.sort_values('Time', inplace=True)
        
        # Reset index
        combined_df = combined_df.reset_index(drop=True)
        
        # Display information about the chamber mappings
        st.subheader("Chamber Mappings")
        for chamber, filename in chamber_mappings.items():
            st.write(f"📊 **{chamber}** - Data from file: {filename}")
        
        # Display combined data
        st.subheader("Combined Data Preview")
        st.write(combined_df.head())
        
        # Get all chamber columns (excluding Time)
        chamber_cols = [col for col in combined_df.columns if col != 'Time']
        
        if chamber_cols:
            # Allow user to select which chambers/values to analyze
            st.subheader("Chamber Selection")
            
            selected_chambers = st.multiselect(
                "Select chambers to analyze", 
                chamber_cols,
                default=chamber_cols  # Default to all chambers
            )
            
            if selected_chambers:
                # Analyze selected chambers
                with st.spinner("Analyzing chamber cycles..."):
                    status.text("Analyzing chamber cycles...")
                    
                    # Ensure Time column is properly formatted
                    combined_df = prepare_dataframe(combined_df)
                    
                    # Analyze each selected chamber
                    results = {}
                    for chamber in selected_chambers:
                        if chamber in combined_df.columns:
                            results[chamber] = analyze_chamber_cycles(combined_df, chamber, start_threshold, end_threshold)
                        else:
                            st.error(f"Selected chamber '{chamber}' not found in combined data.")
                    
                    # Create a list to store cycle data
                    cycle_data = []
                    
                    # Add a debug option to show the analysis process
                    if show_debug_info:
                        show_analysis_steps = st.checkbox("Show Analysis Steps", value=False)
                    else:
                        show_analysis_steps = False
                    
                    # Iterate through results to extract cycle information
                    for chamber, cycles in results.items():
                        if show_analysis_steps:
                            st.write(f"**Analyzing {chamber}:** Found {len(cycles)} cycles")
                            
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
                            
                            if show_analysis_steps:
                                st.write(f"Cycle {i}: Max {cycle['max_value']:.2f} at {cycle['max_time']}, Duration: {cycle['duration']:.2f}s")
                                
                            cycle_data.append(cycle_entry)
                    
                    # Create DataFrame
                    if cycle_data:
                        df_cycles = pd.DataFrame(cycle_data)
                    else:
                        # Create an empty DataFrame with the correct columns if no cycles found
                        df_cycles = pd.DataFrame(columns=[
                            'Chamber', 'Cycle Number', 'Max Value', 'Max Time',
                            'Cycle Duration (seconds)', 'Start Time', 'End Time', 'Peak Time'
                        ])
                    
                    # Display results
                    st.subheader("Chamber Cycle Analysis Results")
                    
                    if df_cycles.empty:
                        st.warning(f"No cycles detected with start threshold {start_threshold} and end threshold {end_threshold}. Try adjusting the thresholds.")
                    else:
                        st.write(df_cycles)
                        
                        # Display summary of cycles per chamber
                        st.subheader("Cycles per Chamber")
                        cycles_summary = df_cycles.groupby('Chamber').size().reset_index(name='Number of Cycles')
                        st.write(cycles_summary)
                        
                        # Plot the chamber fill cycles
                        st.subheader("Chamber Pressure Visualization")
                        
                        # Create a color palette for chambers
                        colors = px.colors.qualitative.Plotly[:len(selected_chambers)]
                        color_map = {chamber: color for chamber, color in zip(selected_chambers, colors)}
                        
                        # Ensure Time column is datetime type before filtering
                        if not pd.api.types.is_datetime64_any_dtype(combined_df['Time']):
                            st.warning("Time column is not in datetime format. Attempting to convert...")
                            combined_df['Time'] = pd.to_datetime(combined_df['Time'], errors='coerce')
                        
                        # Check if we have enough valid timestamps for filtering
                        if combined_df['Time'].isna().sum() > len(combined_df) * 0.5:  # If more than 50% are NaN
                            st.error("Too many invalid timestamps. Time filtering will not work correctly.")
                        
                        min_time = combined_df['Time'].min()
                        max_time = combined_df['Time'].max()
                        
                        if pd.isna(min_time) or pd.isna(max_time):
                            st.error("Could not determine valid time range. Check your data.")
                        elif isinstance(min_time, pd.Timestamp) and isinstance(max_time, pd.Timestamp):
                            # Convert timestamps to strings for display
                            min_str = min_time.strftime('%Y-%m-%d %H:%M:%S')
                            max_str = max_time.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Use date_input instead of slider for time selection
                            col1, col2 = st.columns(2)
                            with col1:
                                start_date = st.date_input("Start date", min_time.date())
                            with col2:
                                end_date = st.date_input("End date", max_time.date())
                            
                            # Convert selected dates to timestamps with time component
                            start_time = pd.Timestamp(start_date)
                            end_time = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                            
                            # Create time range
                            time_range = (start_time, end_time)
                            
                            # Display selected range
                            st.info(f"Selected time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                            
                            # Filter data based on time range
                            filtered_df = combined_df[(combined_df['Time'] >= time_range[0]) & (combined_df['Time'] <= time_range[1])]
                            
                            # Create enhanced visualization
                            st.subheader("Enhanced Chamber Visualization")
                            
                            # Create main figure
                            fig = go.Figure()
                            
                            # Add traces for each chamber
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
                            
                            # Add horizontal lines for thresholds
                            fig.add_shape(
                                type="line",
                                x0=time_range[0],
                                y0=start_threshold,
                                x1=time_range[1],
                                y1=start_threshold,
                                line=dict(color="green", width=2, dash="dash"),
                                name="Start Threshold"
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=time_range[0],
                                y0=end_threshold,
                                x1=time_range[1],
                                y1=end_threshold,
                                line=dict(color="red", width=2, dash="dash"),
                                name="End Threshold"
                            )
                            
                            # Add annotations for thresholds
                            fig.add_annotation(
                                x=time_range[0],
                                y=start_threshold,
                                text=f"Start Threshold: {start_threshold}",
                                showarrow=False,
                                yshift=10,
                                bgcolor="rgba(255,255,255,0.8)"
                            )
                            
                            fig.add_annotation(
                                x=time_range[0],
                                y=end_threshold,
                                text=f"End Threshold: {end_threshold}",
                                showarrow=False,
                                yshift=-20,
                                bgcolor="rgba(255,255,255,0.8)"
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title="Chamber Pressure Over Time",
                                xaxis_title="Time",
                                yaxis_title="Pressure (mmHg)",
                                legend_title="Chambers",
                                height=600,
                                hovermode="x unified",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            # Show the plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add individual chamber plots
                            st.subheader("Individual Chamber Plots")
                            show_individual = st.checkbox("Show Individual Chamber Plots", value=False)
                            
                            if show_individual:
                                # Number of plots per row
                                cols_per_row = 2
                                
                                # Calculate number of rows needed
                                n_rows = (len(selected_chambers) + cols_per_row - 1) // cols_per_row
                                
                                # Create columns
                                for i in range(0, len(selected_chambers), cols_per_row):
                                    # Create row
                                    cols = st.columns(cols_per_row)
                                    
                                    # Add plots to columns
                                    for j in range(cols_per_row):
                                        if i + j < len(selected_chambers):
                                            chamber = selected_chambers[i + j]
                                            with cols[j]:
                                                # Create individual chamber plot
                                                chamber_fig = go.Figure()
                                                
                                                # Add chamber trace
                                                chamber_fig.add_trace(go.Scatter(
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
                                                chamber_fig.add_shape(
                                                    type="line",
                                                    x0=time_range[0],
                                                    y0=start_threshold,
                                                    x1=time_range[1],
                                                    y1=start_threshold,
                                                    line=dict(color="green", width=1, dash="dash")
                                                )
                                                
                                                chamber_fig.add_shape(
                                                    type="line",
                                                    x0=time_range[0],
                                                    y0=end_threshold,
                                                    x1=time_range[1],
                                                    y1=end_threshold,
                                                    line=dict(color="red", width=1, dash="dash")
                                                )
                                                
                                                # Update layout
                                                chamber_fig.update_layout(
                                                    title=chamber,
                                                    height=300,
                                                    margin=dict(l=20, r=20, t=40, b=20),
                                                    xaxis_title=None,
                                                    yaxis_title="Pressure (mmHg)",
                                                    showlegend=False
                                                )
                                                
                                                # Show plot
                                                st.plotly_chart(chamber_fig, use_container_width=True)
                            
                            # Add cycle markers option
                            show_cycles = st.checkbox("Show Cycle Markers on Main Plot", value=True)
                            
                            if show_cycles and not df_cycles.empty:
                                # Create a new figure with cycle markers
                                cycle_fig = go.Figure()
                                
                                # Add traces for each chamber
                                for chamber in selected_chambers:
                                    if chamber in filtered_df.columns:
                                        # Add the main line
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
                                        
                                        # Add cycle peak markers
                                        chamber_cycles = df_cycles[df_cycles['Chamber'] == chamber]
                                        
                                        if not chamber_cycles.empty:
                                            cycle_fig.add_trace(go.Scatter(
                                                x=chamber_cycles['Max Time'],
                                                y=chamber_cycles['Max Value'],
                                                mode='markers',
                                                marker=dict(
                                                    color=color_map.get(chamber, 'blue'),
                                                    size=10,
                                                    symbol='star',
                                                    line=dict(width=2, color='white')
                                                ),
                                                name=f"{chamber} Peaks",
                                                text=[f"Cycle {i}" for i in chamber_cycles['Cycle Number']],
                                                hoverinfo='text+y'
                                            ))
                                
                                # Add threshold lines
                                cycle_fig.add_shape(
                                    type="line",
                                    x0=time_range[0],
                                    y0=start_threshold,
                                    x1=time_range[1],
                                    y1=start_threshold,
                                    line=dict(color="green", width=2, dash="dash")
                                )
                                
                                cycle_fig.add_shape(
                                    type="line",
                                    x0=time_range[0],
                                    y0=end_threshold,
                                    x1=time_range[1],
                                    y1=end_threshold,
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                # Add annotations for thresholds
                                cycle_fig.add_annotation(
                                    x=time_range[0],
                                    y=start_threshold,
                                    text=f"Start Threshold: {start_threshold}",
                                    showarrow=False,
                                    yshift=10,
                                    bgcolor="rgba(255,255,255,0.8)"
                                )
                                
                                cycle_fig.add_annotation(
                                    x=time_range[0],
                                    y=end_threshold,
                                    text=f"End Threshold: {end_threshold}",
                                    showarrow=False,
                                    yshift=-20,
                                    bgcolor="rgba(255,255,255,0.8)"
                                )
                                
                                # Update layout
                                cycle_fig.update_layout(
                                    title="Chamber Pressure Cycles",
                                    xaxis_title="Time",
                                    yaxis_title="Pressure (mmHg)",
                                    legend_title="Chambers",
                                    height=600,
                                    hovermode="closest",
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                # Show the plot
                                st.plotly_chart(cycle_fig, use_container_width=True)
                            
                            # Export options
                            st.subheader("Export Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export combined data
                                combined_csv = combined_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Combined Data (CSV)",
                                    data=combined_csv,
                                    file_name="combined_data.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                # Export cycle analysis
                                cycle_csv = df_cycles.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Cycle Analysis (CSV)",
                                    data=cycle_csv,
                                    file_name="cycle_analysis.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.error("Could not create time range slider due to invalid timestamps in data")
                
                # Progress complete
                progress.progress(100)
                status.text("Analysis complete!")
    
else:
    st.info("Please upload files to begin analysis")
    st.write("""
    ### Instructions:
    1. Upload your chamber data files (Excel or CSV format) using the file uploader in the sidebar
    2. Files will be automatically recognized based on their names (e.g., C1.xls for Chamber 1)
    3. Set the start and end threshold values for cycle detection
    4. Files will be automatically merged on matching timestamps
    5. Select chambers to analyze
    6. View the results and visualizations
    7. Export the data if needed
    
    ### File Naming:
    - Name your files to indicate which chamber they represent (e.g., C1.xls, Chamber2.csv)
    - Each file should contain a Time column and a Value column
    - The analysis tool will automatically extract the chamber number from the filename
    """)
