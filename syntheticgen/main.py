import json
import time
from typing import Dict, Any, List, Union
from llama_cpp import Llama
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
import torch
import numpy as np
import pandas as pd
import warnings
from syntheticgen.config import Config
import requests
from faker import Faker
import re
from datetime import datetime

warnings.filterwarnings("ignore")

class DataGenerator:
    def __init__(self):
        self.local_llm = None
        self.ctgan = None
        self.model_loaded = False
        self.faker = Faker()
        self.field_handlers = {
            'name': self._handle_name_type,
            'int': self._handle_int_type,
            'float': self._handle_float_type,
            'category': self._handle_category_type,
            'date': self._handle_date_type,
            'string': self._handle_string_type,
            'bool': self._handle_bool_type
        }

    def init_local_model(self):
        """Initialize OpenHermes with optimized settings"""
        if not self.model_loaded:
            self.local_llm = Llama(
                model_path=Config.LOCAL_LLM_PATH,
                n_gpu_layers=35,
                n_ctx=2048,
                verbose=False,
                n_threads=4  # Better CPU utilization
            )
            self.local_llm.chat_format = "chatml"  # Force chat mode
            self.model_loaded = True

    def _generate_with_openhermes(self, schema: Dict[str, Any], num_rows: int) -> list:
        """Optimized OpenHermes generation with prompt engineering"""
        self.init_local_model()
        
        prompt = f"""IMPORTANT: You are a JSON data generator. ONLY output JSON.

                  Generate exactly {num_rows} record(s) as a JSON array following these rules:

                  SCHEMA REQUIREMENTS:
                  {json.dumps(schema, indent=2)}

                  STRICT FORMATTING RULES:
                  1. Output MUST be a valid JSON array of objects
                  2. NEVER include code, explanations, or markdown

                  YOUR OUTPUT:"""

        start_time = time.time()
        response = self.local_llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower for consistency
            max_tokens=4000,
            repeat_penalty=1.1  # Reduce repetition
        )
        raw_output = response["choices"][0]["message"]["content"]
        
        # Robust JSON extraction
        json_str = raw_output[raw_output.find('['):raw_output.rfind(']')+1]
        return json.loads(json_str)

    def _generate_with_openrouter(self, schema: Dict[str, Any], num_rows: int) -> list:
        """OpenRouter implementation with adaptive batch sizing"""
        # Start with smaller batches to ensure completion
        initial_batch_size = min(10, num_rows)  # Reduced from 40 to 10
        all_data = []
        remaining_rows = num_rows
        
        while remaining_rows > 0 and len(all_data) < num_rows:
            current_batch_size = min(initial_batch_size, remaining_rows)
            retry_count = 0
            max_retries = 3
            batch_success = False
            
            while retry_count < max_retries and not batch_success:
                try:
                    prompt = f"""Generate exactly {current_batch_size} synthetic records as a JSON array following:
                    {json.dumps(schema, indent=2)}
                    
                    CRITICAL INSTRUCTIONS:
                    1. Output MUST contain EXACTLY {current_batch_size} records
                    2. Format MUST be: [{{"field1":"value1"}}, {{"field2":"value2"}}]
                    3. NEVER include any additional text or explanations
                    4. Strictly validate all values against the schema"""
                    
                    response = requests.post(
                        f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                            # "HTTP-Referer": "https://yourdomain.com",  # Required for some APIs
                            # "X-Title": "Synthetic Data Generator"  # Helps with rate limits
                        },
                        json={
                            "model": Config.OPENROUTER_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "response_format": {"type": "json_object"},
                            "temperature": 0.3,  # Slightly higher for diversity
                            "max_tokens": 4096,  # Maximum allowed for most models
                            "top_p": 0.9
                        },
                        timeout=90  # Increased timeout
                    )
                    
                    # Robust response processing
                    content = response.json()["choices"][0]["message"]["content"]
                    json_str = content[content.find('['):content.rfind(']')+1]
                    batch_data = json.loads(json_str)
                    
                    # Normalize response format
                    if isinstance(batch_data, dict):
                        batch_data = [batch_data]
                    
                    # Validate count and schema
                    valid_records = []
                    for record in batch_data:
                        if all(field in record for field in schema.keys()):
                            valid_records.append(record)
                    
                    if len(valid_records) >= current_batch_size:
                        all_data.extend(valid_records[:current_batch_size])
                        remaining_rows -= current_batch_size
                        batch_success = True
                    else:
                        # If we're short, reduce batch size for next attempt
                        initial_batch_size = max(5, current_batch_size // 2)
                        raise ValueError(f"Got {len(valid_records)} records, expected {current_batch_size}")
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # If failing after retries, try a much smaller batch
                        initial_batch_size = max(5, initial_batch_size // 2)
                    time.sleep(1)  # Brief pause between attempts
        
        # Final validation
        if len(all_data) < num_rows:
            # If we're close, just duplicate some records to reach target
            if len(all_data) > num_rows * 0.9:
                needed = num_rows - len(all_data)
                all_data.extend(all_data[:needed])
            else:
                raise ValueError(f"Only generated {len(all_data)}/{num_rows} rows")
        
        return all_data[:num_rows]

    def _generate_with_ctgan(self, schema: Dict[str, Any], num_rows: int, example_data: List[Dict[str, Any]] = None) -> list:
        """CTGAN generator that respects schema constraints through example data"""
        if not self.ctgan:
            metadata = SingleTableMetadata()
            
            # Add columns with basic types (no constraints in metadata)
            for field, field_type in schema.items():
                base_type = self._get_base_type(field_type)
                if base_type in ['int', 'float']:
                    metadata.add_column(field, sdtype='numerical')
                elif base_type == 'bool':
                    metadata.add_column(field, sdtype='boolean')
                else:
                    metadata.add_column(field, sdtype='categorical')
            
            self.ctgan = CTGANSynthesizer(
                metadata,
                epochs=300,  # Increased epochs for better learning
                batch_size=500,
                cuda=True if torch.cuda.is_available() else False,
                generator_dim=(256, 256, 256),  # Deeper network
                discriminator_dim=(256, 256, 256),
                verbose=True
            )
            
            self.faker.unique.clear()
            
            # Generate or validate example data that respects constraints
            if not example_data:
                example_data = [{
                    field: self._generate_fake_value(field_type)
                    for field, field_type in schema.items()
                } for _ in range(max(20, num_rows//10))]
                
                # Ensure multiple distinct values per field
                for field in schema.keys():
                    values = {x[field] for x in example_data}
                    if len(values) < 3:  # Require at least 3 distinct values
                        new_values = set()
                        while len(new_values) < 3:
                            new_values.add(self._generate_fake_value(schema[field]))
                        for i, value in enumerate(new_values):
                            if i < len(example_data):
                                example_data[i][field] = value
            else:
                # Ensure user-provided data has at least 2 samples
                if len(example_data) < 2:
                    example_data.append({
                        field: self._generate_fake_value(field_type)
                        for field, field_type in schema.items()
                    })
                # Ensure example data respects constraints
                for record in example_data:
                    for field, field_type in schema.items():
                        if field in record:
                            record[field] = self._enforce_constraints(record[field], field_type)
            
            # Convert to DataFrame with proper dtypes
            example_df = pd.DataFrame(example_data)
            for field, field_type in schema.items():
                base_type = self._get_base_type(field_type)
                if base_type == 'int':
                    example_df[field] = pd.to_numeric(example_df[field], errors='coerce').fillna(0).astype(int)
                elif base_type == 'float':
                    example_df[field] = pd.to_numeric(example_df[field], errors='coerce').fillna(0.0)
            
            self.ctgan.fit(example_df)
            
        try:
            # Generate and enforce constraints on output
            synthetic_data = self.ctgan.sample(num_rows * 1.2)  # Generate 20% extra
            synthetic_data = synthetic_data.drop_duplicates()
            synthetic_data = synthetic_data.head(num_rows)
            
            # If we still don't have enough, generate additional unique rows
            if len(synthetic_data) < num_rows:
                additional = self.ctgan.sample(num_rows - len(synthetic_data))
                synthetic_data = pd.concat([synthetic_data, additional]).drop_duplicates().head(num_rows)
                
        except Exception as e:
            # Fallback to basic sampling if enhanced sampling fails
            synthetic_data = self.ctgan.sample(num_rows)
        
        for field, field_type in schema.items():
            if self._get_base_type(field_type) in ['int', 'float']:
                synthetic_data[field] = self._enforce_column_constraints(
                    synthetic_data[field], 
                    field_type
                )
        return synthetic_data.to_dict('records')
        
    def _enforce_constraints(self, value: Any, field_type: str) -> Any:
        """Enforce all field constraints on a single value"""
        base_type = self._get_base_type(field_type)
        
        # Handle integers with min/max constraints
        if base_type == 'int' and ':' in field_type:
            constraints = field_type.split(':')[1]
            if '-' in constraints:
                try:
                    min_val, max_val = map(int, constraints.split('-'))
                    if isinstance(value, (int, float)):
                        return max(min_val, min(max_val, int(value)))
                except (ValueError, TypeError):
                    return self._generate_fake_value(field_type)
        
        # Handle floats with min/max constraints        
        elif base_type == 'float' and ':' in field_type:
            constraints = field_type.split(':')[1]
            if '-' in constraints:
                try:
                    min_val, max_val = map(float, constraints.split('-'))
                    if isinstance(value, (int, float)):
                        return max(min_val, min(max_val, float(value)))
                except (ValueError, TypeError):
                    return self._generate_fake_value(field_type)
        
        # Handle categorical fields
        elif base_type == 'category' and '[' in field_type and ']' in field_type:
            options = field_type.split('[')[1].split(']')[0].split(',')
            options = [opt.strip().strip("'\"") for opt in options if opt.strip()]
            if str(value) not in options and options:
                return self.faker.random_element(options)
        
        # Handle date fields
        elif base_type == 'date':
            try:
                if 'past' in field_type.lower():
                    if pd.to_datetime(value) > datetime.now():
                        return self.faker.past_date().strftime('%Y-%m-%d')
                elif 'future' in field_type.lower():
                    if pd.to_datetime(value) < datetime.now():
                        return self.faker.future_date().strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                return self._generate_fake_value(field_type)
        
        # Handle string patterns
        elif base_type == 'string' and 'pattern=' in field_type:
            pattern = field_type.split('pattern=')[1].split(')')[0].strip('"\'')
            if not re.match(pattern, str(value)):
                return self._generate_from_pattern(pattern)
        
        return value
        
    def _enforce_column_constraints(self, series: pd.Series, field_type: str) -> pd.Series:
        """Apply constraints to an entire pandas Series"""
        base_type = self._get_base_type(field_type)
        
        # Integer constraints
        if base_type == 'int' and ':' in field_type:
            constraints = field_type.split(':')[1]
            if '-' in constraints:
                try:
                    min_val, max_val = map(int, constraints.split('-'))
                    series = pd.to_numeric(series, errors='coerce')
                    return series.clip(lower=min_val, upper=max_val).astype(int)
                except (ValueError, TypeError):
                    pass
        
        # Float constraints
        elif base_type == 'float' and ':' in field_type:
            constraints = field_type.split(':')[1]
            if '-' in constraints:
                try:
                    min_val, max_val = map(float, constraints.split('-'))
                    series = pd.to_numeric(series, errors='coerce')
                    return series.clip(lower=min_val, upper=max_val)
                except (ValueError, TypeError):
                    pass
        
        # Categorical constraints
        elif base_type == 'category' and '[' in field_type and ']' in field_type:
            options = field_type.split('[')[1].split(']')[0].split(',')
            options = [opt.strip().strip("'\"") for opt in options if opt.strip()]
            if options:
                invalid_mask = ~series.isin(options)
                if invalid_mask.any():
                    replacements = [self.faker.random_element(options) for _ in range(invalid_mask.sum())]
                    series = series.mask(invalid_mask, pd.Series(replacements, index=series[invalid_mask].index))
        
        # Date constraints
        elif base_type == 'date':
            try:
                dates = pd.to_datetime(series, errors='coerce')
                if 'past' in field_type.lower():
                    series = dates.where(dates <= datetime.now(), 
                                        pd.Series([self.faker.past_date() for _ in range(len(series))]))
                elif 'future' in field_type.lower():
                    series = dates.where(dates >= datetime.now(), 
                                        pd.Series([self.faker.future_date() for _ in range(len(series))]))
                return series.dt.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                pass
        
        # String pattern constraints
        elif base_type == 'string' and 'pattern=' in field_type:
            pattern = field_type.split('pattern=')[1].split(')')[0].strip('"\'')
            invalid_mask = ~series.astype(str).str.match(pattern, na=False)
            if invalid_mask.any():
                replacements = [self._generate_from_pattern(pattern) for _ in range(invalid_mask.sum())]
                series = series.mask(invalid_mask, pd.Series(replacements, index=series[invalid_mask].index))
        
        if base_type in ['int', 'float']:
            if len(series.unique()) / len(series) < 0.9:  # If low diversity
                noise = pd.Series(np.random.uniform(-0.5, 0.5, size=len(series)))
                series = series + noise
                if base_type == 'int':
                    series = series.round().astype(int)
                    
        return series

    def _get_base_type(self, field_type: str) -> str:
        """Extract base type from field type string"""
        return field_type.split(':')[0].lower() if ':' in field_type else field_type.lower()

    def _generate_fake_value(self, field_type: str) -> Any:
        """Completely generic fake data generation"""
        base_type = self._get_base_type(field_type)
        
        # Special handling for name fields
        if base_type == 'string' and field_type.lower() == 'name':
            return self.faker.unique.name()  # Ensures unique names
        
        handler = self.field_handlers.get(base_type, self._handle_default)
        return handler(None, field_type, None, generate_only=True)
        
    def _handle_name_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, str]:
        """Generate realistic unique names"""
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='categorical')
            return None
        return self.faker.name()

    # Field type handlers
    def _handle_int_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, int]:
        """Handle integer fields with proper constraint parsing"""
        constraints = {}
        if ':' in field_type:
            # Extract the part after the colon (e.g., "18-90" from "int:18-90")
            constraints_str = field_type.split(':')[1]
            
            # Parse range constraints (e.g., "18-90")
            if '-' in constraints_str:
                try:
                    min_val, max_val = map(int, constraints_str.split('-'))
                    constraints['min_value'] = min_val
                    constraints['max_value'] = max_val
                except ValueError:
                    pass
        
        if metadata and not generate_only:
            # SDV expects min_value/max_value for numerical constraints
            metadata.add_column(field, sdtype='numerical', **constraints)
            return None
        
        # Generate actual values within constraints
        min_val = constraints.get('min_value', 0)
        max_val = constraints.get('max_value', 100)
        return self.faker.random_int(min=min_val, max=max_val)

    def _handle_float_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, float]:
        """Handle float fields"""
        constraints = {}
        if ':' in field_type:
            constraints = self._parse_constraints(field_type.split(':')[1])
        
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='numerical', **constraints)
            return None
        
        min_val = constraints.get('min', 0.0)
        max_val = constraints.get('max', 100.0)
        return round(self.faker.random.uniform(min_val, max_val), 2)

    def _handle_category_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, str]:
        """Handle categorical fields"""
        options = []
        if '[' in field_type and ']' in field_type:
            options = field_type.split('[')[1].split(']')[0].split(',')
            options = [opt.strip() for opt in options if opt.strip()]
        
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='categorical')
            return None
        
        return self.faker.random_element(options) if options else self.faker.word()

    def _handle_date_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, str]:
        """Handle date fields"""
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='datetime')
            return None
        
        date_format = '%Y-%m-%d'
        if 'format=' in field_type:
            date_format = field_type.split('format=')[1].split(')')[0].strip('"\'')
        
        if 'past' in field_type.lower():
            return self.faker.past_date().strftime(date_format)
        elif 'future' in field_type.lower():
            return self.faker.future_date().strftime(date_format)
        return self.faker.date_this_year().strftime(date_format)

    def _handle_string_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, str]:
        """Handle string fields"""
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='categorical')
            return None
        
        if 'pattern=' in field_type:
            pattern = field_type.split('pattern=')[1].split(')')[0].strip('"\'')
            return self._generate_from_pattern(pattern)
        return self.faker.word()

    def _handle_bool_type(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, bool]:
        """Handle boolean fields"""
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='boolean')
            return None
        return self.faker.boolean()

    def _handle_default(self, field: str, field_type: str, metadata: SingleTableMetadata = None, generate_only: bool = False) -> Union[None, str]:
        """Default handler for unknown types"""
        if metadata and not generate_only:
            metadata.add_column(field, sdtype='categorical')
            return None
        return self.faker.word()

    def _parse_constraints(self, constraints_str: str) -> Dict[str, Any]:
        """Improved constraint parsing that works with SDV"""
        constraints = {}
        try:
            # Handle range constraints (e.g., "18-90")
            if '-' in constraints_str and all(p.isdigit() for p in constraints_str.split('-')):
                min_val, max_val = map(int, constraints_str.split('-'))
                constraints.update({
                    'min_value': min_val,
                    'max_value': max_val
                })
            # Handle key-value constraints (e.g., "min=18,max=90")
            elif '=' in constraints_str:
                for part in constraints_str.split(','):
                    if '=' in part:
                        key, val = part.split('=', 1)
                        key = key.strip().lower()
                        # Convert numerical constraints
                        if key in ['min', 'max', 'min_value', 'max_value'] and val.isdigit():
                            constraints[f"{key}_value"] = int(val)
                        else:
                            constraints[key] = val.strip(' "\'')
        except Exception:
            pass
        return constraints

    def _generate_from_pattern(self, pattern: str) -> str:
        """Generate data matching a regex pattern"""
        # Simple pattern implementations - can be extended
        if r'\d{4}-\d{2}-\d{2}' in pattern:  # Date pattern
            return datetime.now().strftime('%Y-%m-%d')
        elif r'\d{3}-\d{2}-\d{4}' in pattern:  # SSN pattern
            return self.faker.ssn()
        elif r'\d{3}-\d{3}-\d{4}' in pattern:  # Phone pattern
            return self.faker.phone_number()
        elif r'\d{5}(-\d{4})?' in pattern:  # ZIP code pattern
            return self.faker.zipcode()
        elif r'\d{2,3}/\d{2,3}' in pattern:  # Blood pressure
            return f"{self.faker.random_int(90,180)}/{self.faker.random_int(60,120)}"
        # Fallback to simple random string
        return ''.join(self.faker.random_letters(length=8))

    def generate_batch(self, schema: Dict[str, Any], batch_size: int, engine: str = "openhermes", example_data: List[Dict[str, Any]] = None) -> list:
        """Router for all generation methods"""
        if engine == "openhermes":
            return self._generate_with_openhermes(schema, batch_size)
        elif engine == "openrouter":
            return self._generate_with_openrouter(schema, batch_size)
        elif engine == "ctgan":
            return self._generate_with_ctgan(schema, batch_size, example_data)
        raise ValueError(f"Unknown engine: {engine}")