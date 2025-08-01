import requests
import streamlit as st
import pandas as pd
import json
import time
from syntheticgen.main import DataGenerator

def main():
    generator = DataGenerator()
    st.set_page_config(layout="wide", page_title="Synthetic Data Generator", page_icon=":infinity:")

    # Sidebar Controls
    with st.sidebar:
        st.header("Engine Configuration")
        engine = st.radio(
            "Generation Engine",
            ["openhermes", "openrouter", "ctgan"],
            index=0,
            format_func=lambda x: {
                "openhermes": "🚀 OpenHermes (Local)",
                "openrouter": "🌐 OpenRouter (API)",
                "ctgan": "📊 CTGAN (Tabular)"
            }[x]
        )
        
        # Common parameters
        num_rows = st.number_input(
            "Number of Rows",
            min_value=1,
            max_value=1000 if engine == "openhermes" else 1000000,
            value=5,
            step=1
        )
        
        # Engine-specific notes
        if engine == "openhermes":
            st.warning("OpenHermes speed: ~1-2 min per row")
        elif engine == "ctgan":
            st.info("CTGAN: Provide example data for better results")
        
        # CTGAN-specific controls
        example_data = None
        if engine == "ctgan":
            example_file = st.file_uploader(
                "Upload Example Data (JSON)",
                type=["json"],
                help="Optional but recommended for better CTGAN results"
            )
            if example_file:
                try:
                    example_data = json.load(example_file)
                    if not isinstance(example_data, list):
                        example_data = [example_data]  # Convert single record to list
                    st.success(f"Loaded {len(example_data)} example records")
                except Exception as e:
                    st.error(f"Error loading example data: {str(e)}")

    # Main Interface
    st.title("Multi-model Synthetic Data Generator")
    uploaded_file = st.file_uploader("Upload Schema (JSON)", type=["json"])

    if uploaded_file and st.button("Generate", type="primary"):
        schema = json.load(uploaded_file)
        
        with st.spinner(f"Generating {num_rows} rows via {engine}..."):
            try:
                start_time = time.time()
                
                # Handle CTGAN case with example data
                if engine == "ctgan":
                    data = generator.generate_batch(
                        schema=schema,
                        batch_size=num_rows,
                        engine=engine,
                        example_data=example_data
                    )
                elif engine == "openrouter":
                    # Create progress elements
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    progress_container = st.empty()
                    
                    # Initialize variables
                    all_data = []
                    batch_size = 10 if num_rows > 10 else num_rows
                    total_batches = (num_rows + batch_size - 1) // batch_size
                    
                    for batch_num in range(total_batches):
                        current_batch_size = min(batch_size, num_rows - len(all_data))
                        
                        # Update progress
                        progress = (batch_num + 1) / total_batches
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Generating batch {batch_num+1}/{total_batches}...")
                        
                        # Generate batch
                        batch_data = generator._generate_with_openrouter(
                            schema=schema,
                            num_rows=current_batch_size
                        )
                        all_data.extend(batch_data)
                        
                        # Show batch stats
                        progress_container.text(f"""
                            Batch {batch_num+1} complete
                            Records generated: {len(all_data)}/{num_rows}
                            Estimated time remaining: {(time.time() - start_time) * (total_batches - batch_num - 1) / (batch_num + 1):.1f}s
                        """)
                    
                    data = all_data[:num_rows]  # Ensure exact count
                else:
                    data = generator.generate_batch(
                        schema=schema,
                        batch_size=num_rows,
                        engine=engine
                    )
                
                gen_time = time.time() - start_time
                
                df = pd.DataFrame(data)
                st.success(f"✅ Generated {len(df)} rows in {gen_time:.1f} seconds")

                if data:
                    # Results Display
                    st.dataframe(df.head(10))
                    
                    # Performance notes
                    if engine == "ctgan":
                        st.info(f"CTGAN speed: {num_rows/gen_time:.0f} rows/second")
                    
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        "synthetic_data.csv",
                        "text/csv"
                    )
                else:
                    st.error("No data was generated - please check the schema and try again")
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.json({"error": str(e), "schema": schema})  # Debug info

if __name__ == "__main__":
    main()