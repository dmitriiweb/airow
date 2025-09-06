import pytest
from pydantic import BaseModel, ValidationError

from airow.agent import AirowAgent
from airow.schemas import OutputColumn


def test_single_output_column():
    """Test creating a model with a single output column."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="result", type=str, description="The result of processing")
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Verify it's a BaseModel subclass
    assert issubclass(model_class, BaseModel)
    
    # Verify the model name
    assert model_class.__name__ == "OutputColumns"
    
    # Verify field exists
    assert "result" in model_class.model_fields
    
    # Verify field configuration
    field_info = model_class.model_fields["result"]
    assert field_info.annotation is str
    assert field_info.description == "The result of processing"
    assert field_info.is_required()


def test_multiple_output_columns():
    """Test creating a model with multiple output columns."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="name", type=str, description="Person's name"),
        OutputColumn(name="age", type=int, description="Person's age"),
        OutputColumn(name="is_active", type=bool, description="Whether person is active"),
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Verify all fields exist
    assert "name" in model_class.model_fields
    assert "age" in model_class.model_fields
    assert "is_active" in model_class.model_fields
    
    # Verify field types and descriptions
    name_field = model_class.model_fields["name"]
    assert name_field.annotation is str
    assert name_field.description == "Person's name"
    
    age_field = model_class.model_fields["age"]
    assert age_field.annotation is int
    assert age_field.description == "Person's age"
    
    is_active_field = model_class.model_fields["is_active"]
    assert is_active_field.annotation is bool
    assert is_active_field.description == "Whether person is active"


def test_different_data_types():
    """Test creating a model with various data types."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="text", type=str, description="Text field"),
        OutputColumn(name="number", type=int, description="Integer field"),
        OutputColumn(name="float_val", type=float, description="Float field"),
        OutputColumn(name="flag", type=bool, description="Boolean field"),
        OutputColumn(name="items", type=list, description="List field"),
        OutputColumn(name="metadata", type=dict, description="Dictionary field"),
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Verify all fields exist with correct types
    assert model_class.model_fields["text"].annotation is str
    assert model_class.model_fields["number"].annotation is int
    assert model_class.model_fields["float_val"].annotation is float
    assert model_class.model_fields["flag"].annotation is bool
    assert model_class.model_fields["items"].annotation is list
    assert model_class.model_fields["metadata"].annotation is dict


def test_empty_output_columns():
    """Test creating a model with no output columns."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = []
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Should still create a valid model class
    assert issubclass(model_class, BaseModel)
    assert model_class.__name__ == "OutputColumns"
    assert len(model_class.model_fields) == 0


def test_model_validation():
    """Test that the created model validates data correctly."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="name", type=str, description="Person's name"),
        OutputColumn(name="age", type=int, description="Person's age"),
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Test valid data
    valid_data = {"name": "John Doe", "age": 30}
    instance = model_class(**valid_data)
    assert instance.name == "John Doe"
    assert instance.age == 30
    
    # Test invalid data (missing required field)
    with pytest.raises(ValidationError):
        model_class(name="John Doe")  # Missing age
    
    # Test invalid data (wrong type)
    with pytest.raises(ValidationError):
        model_class(name="John Doe", age="thirty")  # age should be int


def test_model_serialization():
    """Test that the created model can be serialized."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="result", type=str, description="Processing result"),
        OutputColumn(name="score", type=float, description="Processing score"),
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    instance = model_class(result="success", score=0.95)
    
    # Test model_dump
    data = instance.model_dump()
    expected = {"result": "success", "score": 0.95}
    assert data == expected
    
    # Test model_dump_json
    json_data = instance.model_dump_json()
    assert '"result":"success"' in json_data
    assert '"score":0.95' in json_data


def test_field_descriptions_preserved():
    """Test that field descriptions are properly preserved."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(
            name="complex_field", 
            type=str, 
            description="This is a complex field with special characters: @#$%^&*()"
        ),
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    field_info = model_class.model_fields["complex_field"]
    assert field_info.description == "This is a complex field with special characters: @#$%^&*()"


def test_duplicate_field_names():
    """Test behavior with duplicate field names (should overwrite)."""
    agent = AirowAgent(model=None, system_prompt="test")
    
    output_columns = [
        OutputColumn(name="field", type=str, description="First field"),
        OutputColumn(name="field", type=int, description="Second field"),  # Duplicate name
    ]
    
    model_class = agent.build_agent_output_type(output_columns)
    
    # Should have only one field (last one wins)
    assert len(model_class.model_fields) == 1
    assert "field" in model_class.model_fields
    
    # Should use the last definition
    field_info = model_class.model_fields["field"]
    assert field_info.annotation is int
    assert field_info.description == "Second field"