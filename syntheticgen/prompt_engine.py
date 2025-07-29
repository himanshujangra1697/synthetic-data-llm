from typing import Dict, Any
import json

# prompt_engine.py
class PromptEngine:
    def __init__(self, schema: list[Dict[str, Any]]):
        self.schema = schema
    
    def generate_prompt(self, num_rows: int) -> str:
        # Build context-aware prompt based on schema
        constraints = []
        for field in self.schema:
            if field.type == "int":
                c = f"{field.name}: integer"
                if field.constraints.min or field.constraints.max:
                    c += f" between {field.constraints.min}-{field.constraints.max}"
            # Add other types...
            constraints.append(c)
        
        return f"""
        Generate {num_rows} synthetic records as JSON array.
        Strict schema:
        {json.dumps([f.dict() for f in self.schema], indent=2)}
        
        Output ONLY JSON array with no additional text.
        Example: [{{"field1": value, "field2": value}}]
        """