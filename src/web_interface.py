#!/usr/bin/env python3
"""
Web Interface for Interactive Fraud Detection Testing

Provides a FastAPI web interface to test the fraud detection system
with custom transaction inputs.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import json
import uvicorn

from .agents.pattern_recognition_agent import PatternRecognitionAgent
from .agents.risk_assessment_agent import RiskAssessmentAgent
from .agents.coordination_agent import CoordinationAgent
from .utils.synthetic_data import SyntheticDataGenerator
from .schemas.transaction_schema import TransactionSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic AI Fraud Detection System",
    description="Interactive web interface for testing fraud detection with custom transactions",
    version="1.0.0"
)

# Global system components
pattern_agent = None
risk_agent = None
coordination_agent = None
data_generator = None

class TransactionInput(BaseModel):
    """Input model for custom transactions"""
    user_id: str = "test_user"
    amount: float
    currency: str = "USD"
    transaction_type: str = "purchase"
    merchant_category: Optional[str] = "retail"
    country: Optional[str] = "US"
    timestamp: Optional[str] = None
    device_type: Optional[str] = "mobile"
    is_weekend: Optional[bool] = None
    hour_of_day: Optional[int] = None

class FraudTestResponse(BaseModel):
    """Response model for fraud detection results"""
    transaction_id: str
    fraud_score: float
    decision: str
    confidence: float
    risk_factors: list
    processing_time_ms: float
    agent_analysis: dict
    explanation: str

@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection system"""
    global pattern_agent, risk_agent, coordination_agent, data_generator
    
    logger.info("🚀 Initializing Fraud Detection System...")
    
    # Initialize agents
    pattern_agent = PatternRecognitionAgent()
    risk_agent = RiskAssessmentAgent()
    coordination_agent = CoordinationAgent()
    data_generator = SyntheticDataGenerator()
    
    # Register agents
    await coordination_agent.register_agent(pattern_agent)
    await coordination_agent.register_agent(risk_agent)
    
    logger.info("✅ Fraud Detection System initialized!")

@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Agentic AI Fraud Detection System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                color: #4a5568;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .subtitle {
                text-align: center;
                color: #718096;
                margin-bottom: 40px;
                font-size: 1.2em;
            }
            .form-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            .form-group {
                display: flex;
                flex-direction: column;
            }
            label {
                font-weight: 600;
                margin-bottom: 5px;
                color: #4a5568;
            }
            input, select {
                padding: 12px;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            .btn-container {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin: 30px 0;
            }
            button {
                padding: 15px 30px;
                font-size: 16px;
                font-weight: 600;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .btn-primary {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .btn-secondary {
                background: #e2e8f0;
                color: #4a5568;
            }
            .btn-secondary:hover {
                background: #cbd5e0;
            }
            .results {
                margin-top: 30px;
                padding: 20px;
                background: #f7fafc;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }
            .metric {
                display: inline-block;
                margin: 10px 15px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
                min-width: 120px;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: 700;
                margin-bottom: 5px;
            }
            .metric-label {
                color: #718096;
                font-size: 0.9em;
            }
            .fraud-high { color: #e53e3e; }
            .fraud-medium { color: #d69e2e; }
            .fraud-low { color: #38a169; }
            .explanation {
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #4299e1;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .examples {
                margin: 30px 0;
                text-align: center;
            }
            .example-btn {
                margin: 5px;
                padding: 8px 16px;
                background: #edf2f7;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            }
            .example-btn:hover {
                background: #e2e8f0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Agentic AI Fraud Detection</h1>
            <p class="subtitle">Test custom transactions with our multi-agent AI system</p>
            
            <form id="fraudForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="amount">Transaction Amount ($)</label>
                        <input type="number" id="amount" step="0.01" min="0" value="100.00" required>
                    </div>
                    <div class="form-group">
                        <label for="merchant_category">Merchant Category</label>
                        <select id="merchant_category">
                            <option value="retail">Retail</option>
                            <option value="grocery">Grocery</option>
                            <option value="gas">Gas Station</option>
                            <option value="restaurant">Restaurant</option>
                            <option value="online">Online</option>
                            <option value="gambling">Gambling</option>
                            <option value="cryptocurrency">Cryptocurrency</option>
                            <option value="money_transfer">Money Transfer</option>
                            <option value="adult">Adult Services</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="country">Country</label>
                        <select id="country">
                            <option value="US">United States</option>
                            <option value="CA">Canada</option>
                            <option value="GB">United Kingdom</option>
                            <option value="DE">Germany</option>
                            <option value="FR">France</option>
                            <option value="RU">Russia</option>
                            <option value="NG">Nigeria</option>
                            <option value="PK">Pakistan</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="hour_of_day">Hour of Day (0-23)</label>
                        <input type="number" id="hour_of_day" min="0" max="23" value="14">
                    </div>
                    <div class="form-group">
                        <label for="device_type">Device Type</label>
                        <select id="device_type">
                            <option value="mobile">Mobile</option>
                            <option value="desktop">Desktop</option>
                            <option value="tablet">Tablet</option>
                            <option value="">Unknown</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="user_id">User ID</label>
                        <input type="text" id="user_id" value="test_user_001">
                    </div>
                </div>
                
                <div class="examples">
                    <h3>Quick Examples:</h3>
                    <button type="button" class="example-btn" onclick="loadExample('normal')">Normal Transaction</button>
                    <button type="button" class="example-btn" onclick="loadExample('high_amount')">High Amount</button>
                    <button type="button" class="example-btn" onclick="loadExample('late_night')">Late Night</button>
                    <button type="button" class="example-btn" onclick="loadExample('crypto')">Cryptocurrency</button>
                    <button type="button" class="example-btn" onclick="loadExample('high_risk')">High Risk Country</button>
                </div>
                
                <div class="btn-container">
                    <button type="submit" class="btn-primary">🔍 Analyze Transaction</button>
                    <button type="button" class="btn-secondary" onclick="generateRandom()">🎲 Random Transaction</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>AI agents are analyzing your transaction...</p>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <h3>🎯 Fraud Detection Results</h3>
                <div id="metrics"></div>
                <div id="explanation" class="explanation"></div>
            </div>
        </div>

        <script>
            const examples = {
                normal: { amount: 45.99, merchant_category: 'grocery', country: 'US', hour_of_day: 14, device_type: 'mobile' },
                high_amount: { amount: 8500.00, merchant_category: 'retail', country: 'US', hour_of_day: 15, device_type: 'desktop' },
                late_night: { amount: 250.00, merchant_category: 'online', country: 'US', hour_of_day: 2, device_type: 'mobile' },
                crypto: { amount: 5000.00, merchant_category: 'cryptocurrency', country: 'US', hour_of_day: 16, device_type: 'desktop' },
                high_risk: { amount: 1200.00, merchant_category: 'money_transfer', country: 'NG', hour_of_day: 20, device_type: '' }
            };

            function loadExample(type) {
                const example = examples[type];
                Object.keys(example).forEach(key => {
                    const element = document.getElementById(key);
                    if (element) element.value = example[key];
                });
            }

            function generateRandom() {
                const amounts = [25.50, 89.99, 156.78, 899.00, 2500.00, 7800.00, 15000.00];
                const hours = [2, 8, 12, 14, 18, 22, 23];
                
                document.getElementById('amount').value = amounts[Math.floor(Math.random() * amounts.length)];
                document.getElementById('hour_of_day').value = hours[Math.floor(Math.random() * hours.length)];
            }

            document.getElementById('fraudForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                
                loading.style.display = 'block';
                results.style.display = 'none';
                
                const formData = {
                    amount: parseFloat(document.getElementById('amount').value),
                    merchant_category: document.getElementById('merchant_category').value,
                    country: document.getElementById('country').value,
                    hour_of_day: parseInt(document.getElementById('hour_of_day').value),
                    device_type: document.getElementById('device_type').value || null,
                    user_id: document.getElementById('user_id').value
                };
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    displayResults(result);
                } catch (error) {
                    alert('Error analyzing transaction: ' + error.message);
                } finally {
                    loading.style.display = 'none';
                }
            });

            function displayResults(result) {
                const results = document.getElementById('results');
                const metrics = document.getElementById('metrics');
                const explanation = document.getElementById('explanation');
                
                const fraudClass = result.fraud_score > 0.7 ? 'fraud-high' : 
                                 result.fraud_score > 0.4 ? 'fraud-medium' : 'fraud-low';
                
                metrics.innerHTML = `
                    <div class="metric">
                        <div class="metric-value ${fraudClass}">${(result.fraud_score * 100).toFixed(1)}%</div>
                        <div class="metric-label">Fraud Score</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${result.decision.toUpperCase()}</div>
                        <div class="metric-label">Decision</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(result.confidence * 100).toFixed(1)}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${result.processing_time_ms.toFixed(1)}ms</div>
                        <div class="metric-label">Processing Time</div>
                    </div>
                `;
                
                explanation.innerHTML = `
                    <h4>🧠 AI Agent Analysis</h4>
                    <p><strong>Explanation:</strong> ${result.explanation}</p>
                    <p><strong>Risk Factors:</strong> ${result.risk_factors.join(', ') || 'None detected'}</p>
                    <p><strong>Strategy Used:</strong> ${result.agent_analysis.coordination_strategy}</p>
                    <p><strong>Agents Participating:</strong> ${result.agent_analysis.participating_agents.length}</p>
                `;
                
                results.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze", response_model=FraudTestResponse)
async def analyze_transaction(transaction_input: TransactionInput):
    """Analyze a custom transaction for fraud"""
    try:
        start_time = datetime.utcnow()
        
        # Create transaction data
        if transaction_input.timestamp is None:
            timestamp = datetime.utcnow()
            if transaction_input.hour_of_day is not None:
                timestamp = timestamp.replace(hour=transaction_input.hour_of_day, minute=0, second=0)
        else:
            timestamp = datetime.fromisoformat(transaction_input.timestamp)
        
        transaction_data = {
            "transaction_id": f"custom_{int(datetime.utcnow().timestamp())}",
            "user_id": transaction_input.user_id,
            "amount": transaction_input.amount,
            "currency": transaction_input.currency,
            "transaction_type": transaction_input.transaction_type,
            "merchant_category": transaction_input.merchant_category,
            "location": {"country": transaction_input.country} if transaction_input.country else None,
            "timestamp": timestamp.isoformat(),
            "device_info": {"type": transaction_input.device_type} if transaction_input.device_type else None,
            "status": "pending"
        }
        
        # Process through coordination agent
        coord_input = {
            "transaction": transaction_data,
            "task_type": "fraud_detection"
        }
        
        coord_result = await coordination_agent.process(coord_input)
        coordinated_decision = coord_result.get("coordinated_decision", {})
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Extract risk factors from agent results
        risk_factors = []
        agent_results = coord_result.get("agent_results", {})
        
        for agent_id, agent_result in agent_results.items():
            if "risk_factors" in agent_result:
                risk_factors.extend(agent_result["risk_factors"])
            if "new_patterns" in agent_result:
                for pattern in agent_result.get("new_patterns", []):
                    risk_factors.extend(pattern.get("features", {}).keys())
        
        # Generate explanation
        explanation = await _generate_explanation(transaction_data, coordinated_decision, coord_result)
        
        return FraudTestResponse(
            transaction_id=transaction_data["transaction_id"],
            fraud_score=coordinated_decision.get("fraud_score", 0.0),
            decision=coordinated_decision.get("decision", "approve"),
            confidence=coordinated_decision.get("confidence", 0.0),
            risk_factors=list(set(risk_factors)),
            processing_time_ms=processing_time,
            agent_analysis={
                "coordination_strategy": coord_result.get("coordination_strategy", "unknown"),
                "participating_agents": coord_result.get("participating_agents", []),
                "agent_results": agent_results
            },
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Transaction analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/generate-sample")
async def generate_sample_transaction(fraud: bool = False):
    """Generate a sample transaction for testing"""
    if fraud:
        transaction = data_generator.generate_fraudulent_transaction()
    else:
        transaction = data_generator.generate_normal_transaction()
    
    return transaction

@app.get("/system-status")
async def get_system_status():
    """Get current system status and metrics"""
    try:
        pattern_metrics = await pattern_agent.get_performance_metrics()
        risk_metrics = await risk_agent.get_performance_metrics()
        coord_metrics = await coordination_agent.get_performance_metrics()
        
        return {
            "status": "healthy",
            "agents": {
                "pattern_recognition": pattern_metrics,
                "risk_assessment": risk_metrics,
                "coordination": coord_metrics
            },
            "uptime": "running"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def _generate_explanation(transaction_data: Dict[str, Any], decision: Dict[str, Any], coord_result: Dict[str, Any]) -> str:
    """Generate human-readable explanation of the fraud decision"""
    amount = transaction_data.get("amount", 0)
    country = transaction_data.get("location", {}).get("country", "unknown")
    merchant = transaction_data.get("merchant_category", "unknown")
    fraud_score = decision.get("fraud_score", 0)
    
    explanation = f"Transaction of ${amount:,.2f} from {country} at {merchant} merchant. "
    
    if fraud_score > 0.8:
        explanation += "HIGH RISK: Multiple fraud indicators detected. "
    elif fraud_score > 0.5:
        explanation += "MEDIUM RISK: Some suspicious patterns identified. "
    else:
        explanation += "LOW RISK: Transaction appears normal. "
    
    strategy = coord_result.get("coordination_strategy", "unknown")
    agents = len(coord_result.get("participating_agents", []))
    
    explanation += f"Analysis performed by {agents} AI agents using {strategy} coordination strategy."
    
    return explanation

if __name__ == "__main__":
    uvicorn.run(
        "src.web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )