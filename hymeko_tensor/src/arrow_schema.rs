use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Schema for 3D Star/Clique Expansions
pub fn schema_expansion_3d() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("i", DataType::Int64, false),
        Field::new("j", DataType::Int64, false),
        Field::new("val", DataType::Float32, false),
    ]))
}

/// Schema for 2D Projected Expansions
pub fn schema_expansion_2d() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("j", DataType::Int64, false),
        Field::new("val", DataType::Float32, false),
    ]))
}
