def test_product_metadata():
    metadata = ProductMetadata(
        name="Test Product",
        description="A test description",
        specifications={
            "color": "black",
            "size": "medium"
        },
        category="test"
    )
    
    flattened = metadata.flatten_metadata()
    assert flattened['name'] == "Test Product"
    assert flattened['spec_color'] == "black"
    assert flattened['spec_size'] == "medium" 