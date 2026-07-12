def calculate_risk_level(prediction: str, confidence: float) -> str:
    """
    Converts prediction and confidence into a standardized severity level.
    Levels: Low, Medium, High, Critical
    """
    prediction = str(prediction).lower()
    
    if "normal" in prediction or "benign" in prediction:
        return "Low"
        
    # Categorize based on attack type severity
    critical_attacks = ["bot", "ddos", "dos", "heartbleed", "infiltration"]
    high_attacks = ["brute force", "sql injection", "web attack"]
    medium_attacks = ["portscan"]
    
    if any(attack in prediction for attack in critical_attacks):
        severity = "Critical"
    elif any(attack in prediction for attack in high_attacks):
        severity = "High"
    elif any(attack in prediction for attack in medium_attacks):
        severity = "Medium"
    else:
        severity = "High" # Default for unknown attacks
        
    # Adjust severity based on confidence
    # If the model is very unsure about an attack, downgrade the severity
    if severity == "Critical" and confidence < 50.0:
        severity = "High"
    elif severity == "High" and confidence < 50.0:
        severity = "Medium"
        
    return severity
