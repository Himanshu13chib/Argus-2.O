"""
Predictive Analytics Service for Project Argus
Provides historical pattern analysis, anomaly detection, and predictive maintenance
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

print("Starting imports...")

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.security import HTTPBearer
    import uvicorn
    from pydantic import BaseModel
    print("FastAPI imports successful")
except ImportError as e:
    print(f"FastAPI import error: {e}")

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    import joblib
    print("ML library imports successful")
except ImportError as e:
    print(f"ML library import error: {e}")

import json
from dataclasses import dataclass, asdict

print("All imports completed")

# from shared.interfaces.security import SecurityManager
# from shared.models.detection import Detection
# from shared.models.alerts import Alert
# from shared.models.tracking import Track

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predictive Analytics Service", version="1.0.0")
security = HTTPBearer()

@dataclass
class PredictionResult:
    """Result of predictive analysis"""
    prediction_type: str
    confidence: float
    predicted_value: float
    risk_level: str
    recommendations: List[str]
    timestamp: datetime

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    description: str
    severity: str
    timestamp: datetime

@dataclass
class PatternAnalysis:
    """Historical pattern analysis result"""
    pattern_type: str
    frequency: str
    peak_times: List[str]
    seasonal_trends: Dict[str, float]
    correlation_factors: Dict[str, float]

class PredictiveAnalyticsEngine:
    """Main predictive analytics engine"""
    
    def __init__(self):
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize predictive models"""
        # Crossing prediction model
        self.models['crossing_prediction'] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        
        # Hardware failure prediction model
        self.models['hardware_failure'] = RandomForestRegressor(
            n_estimators=50, random_state=42
        )
        
        # Anomaly detection models
        self.anomaly_detectors['crossing_patterns'] = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.anomaly_detectors['behavioral'] = IsolationForest(
            contamination=0.05, random_state=42
        )
        
        # Scalers for data normalization
        self.scalers['crossing_features'] = StandardScaler()
        self.scalers['hardware_metrics'] = StandardScaler()
        self.scalers['environmental'] = StandardScaler()
        
        # Initialize historical data storage
        self.historical_data['crossings'] = pd.DataFrame()
        self.historical_data['hardware_health'] = pd.DataFrame()
        self.historical_data['environmental'] = pd.DataFrame()
        
        # Initialize models with synthetic training data
        self._initialize_with_synthetic_data()
        
        logger.info("Predictive analytics models initialized")
    
    def _initialize_with_synthetic_data(self):
        """Initialize models with synthetic training data for immediate use"""
        # Generate synthetic crossing features for anomaly detector training
        np.random.seed(42)  # For reproducible results
        
        # Normal crossing patterns (majority of data)
        normal_features = []
        for _ in range(100):
            # Normal crossing times (dawn/dusk hours more common)
            hour = np.random.choice([6, 7, 8, 17, 18, 19], p=[0.2, 0.3, 0.2, 0.1, 0.1, 0.1])
            if np.random.random() < 0.3:  # Some random hours
                hour = np.random.randint(0, 24)
            
            feature_vector = [
                hour,  # hour
                np.random.randint(0, 7),  # day of week
                np.random.uniform(0.7, 0.95),  # confidence
                np.random.choice([1, 2], p=[0.8, 0.2]),  # group_size (mostly single)
                np.random.uniform(0.8, 2.0),  # speed
                np.random.uniform(0.85, 1.0)  # direction_consistency
            ]
            normal_features.append(feature_vector)
        
        # Add some anomalous patterns for training
        anomalous_features = []
        for _ in range(10):
            feature_vector = [
                np.random.choice([2, 3, 23]),  # unusual hours
                np.random.randint(0, 7),
                np.random.uniform(0.6, 0.8),  # lower confidence
                np.random.choice([3, 4, 5]),  # large groups
                np.random.uniform(2.5, 4.0),  # fast movement
                np.random.uniform(0.3, 0.7)  # inconsistent direction
            ]
            anomalous_features.append(feature_vector)
        
        # Combine and train
        all_features = np.array(normal_features + anomalous_features)
        
        # Fit scaler and anomaly detector
        self.scalers['crossing_features'].fit(all_features)
        normalized_features = self.scalers['crossing_features'].transform(all_features)
        self.anomaly_detectors['crossing_patterns'].fit(normalized_features)
        
        # Initialize behavioral anomaly detector with similar data
        behavioral_features = []
        for _ in range(80):
            # Normal behavioral patterns
            feature_vector = [
                np.random.uniform(0.5, 2.0),  # speed
                np.random.uniform(0.8, 1.0),  # direction consistency
                np.random.uniform(10, 300),   # time in area (seconds)
                np.random.uniform(0.0, 0.2),  # loitering score
                np.random.uniform(0.7, 0.95)  # confidence
            ]
            behavioral_features.append(feature_vector)
        
        # Add anomalous behavioral patterns
        for _ in range(20):
            feature_vector = [
                np.random.uniform(0.1, 0.4),  # very slow (loitering)
                np.random.uniform(0.3, 0.7),  # erratic movement
                np.random.uniform(600, 1800), # long time in area
                np.random.uniform(0.6, 1.0),  # high loitering score
                np.random.uniform(0.5, 0.8)   # lower confidence
            ]
            behavioral_features.append(feature_vector)
        
        behavioral_array = np.array(behavioral_features)
        self.anomaly_detectors['behavioral'].fit(behavioral_array)
        
        logger.info("Models initialized with synthetic training data")
    
    async def analyze_historical_patterns(self, camera_id: str, days_back: int = 30) -> PatternAnalysis:
        """Analyze historical crossing patterns for anomaly detection (Requirement 8.1)"""
        try:
            # Get historical crossing data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Simulate historical data retrieval (in production, this would query the database)
            crossing_data = await self._get_historical_crossings(camera_id, start_date, end_date)
            
            if crossing_data.empty:
                return PatternAnalysis(
                    pattern_type="insufficient_data",
                    frequency="unknown",
                    peak_times=[],
                    seasonal_trends={},
                    correlation_factors={}
                )
            
            # Analyze temporal patterns
            crossing_data['hour'] = pd.to_datetime(crossing_data['timestamp']).dt.hour
            crossing_data['day_of_week'] = pd.to_datetime(crossing_data['timestamp']).dt.dayofweek
            crossing_data['month'] = pd.to_datetime(crossing_data['timestamp']).dt.month
            
            # Calculate frequency patterns
            hourly_counts = crossing_data.groupby('hour').size()
            daily_counts = crossing_data.groupby('day_of_week').size()
            monthly_counts = crossing_data.groupby('month').size()
            
            # Identify peak times
            peak_hours = hourly_counts.nlargest(3).index.tolist()
            peak_times = [f"{hour:02d}:00" for hour in peak_hours]
            
            # Calculate seasonal trends
            seasonal_trends = {
                'morning_activity': float(hourly_counts[6:12].mean()),
                'afternoon_activity': float(hourly_counts[12:18].mean()),
                'evening_activity': float(hourly_counts[18:24].mean()),
                'night_activity': float(hourly_counts[0:6].mean())
            }
            
            # Calculate correlation factors
            correlation_factors = {
                'weekday_vs_weekend': float(daily_counts[:5].mean() / daily_counts[5:].mean()) if daily_counts[5:].mean() > 0 else 1.0,
                'peak_hour_concentration': float(hourly_counts.max() / hourly_counts.mean()) if hourly_counts.mean() > 0 else 1.0
            }
            
            # Determine overall frequency
            avg_daily_crossings = len(crossing_data) / days_back
            if avg_daily_crossings > 10:
                frequency = "high"
            elif avg_daily_crossings > 3:
                frequency = "medium"
            else:
                frequency = "low"
            
            return PatternAnalysis(
                pattern_type="temporal_analysis",
                frequency=frequency,
                peak_times=peak_times,
                seasonal_trends=seasonal_trends,
                correlation_factors=correlation_factors
            )
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")
    
    async def detect_anomalous_behavior(self, recent_crossings: List[Dict]) -> AnomalyDetection:
        """Detect anomalous crossing behavior (Requirement 8.1)"""
        try:
            if not recent_crossings:
                return AnomalyDetection(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    anomaly_type="no_data",
                    description="No recent crossing data available",
                    severity="low",
                    timestamp=datetime.now()
                )
            
            # Extract features for anomaly detection
            features = []
            for crossing in recent_crossings:
                crossing_time = pd.to_datetime(crossing['timestamp'])
                feature_vector = [
                    crossing_time.hour,
                    crossing_time.dayofweek,
                    crossing['confidence'],
                    crossing.get('group_size', 1),
                    crossing.get('speed', 0.0),
                    crossing.get('direction_consistency', 1.0)
                ]
                features.append(feature_vector)
            
            features_array = np.array(features)
            
            # Normalize features
            if not hasattr(self.scalers['crossing_features'], 'mean_'):
                # If scaler not fitted, fit with current data
                self.scalers['crossing_features'].fit(features_array)
            
            normalized_features = self.scalers['crossing_features'].transform(features_array)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detectors['crossing_patterns'].decision_function(normalized_features)
            anomaly_predictions = self.anomaly_detectors['crossing_patterns'].predict(normalized_features)
            
            # Calculate overall anomaly score
            avg_anomaly_score = float(np.mean(anomaly_scores))
            is_anomaly = bool(np.any(anomaly_predictions == -1))
            
            # Determine anomaly type and severity
            if is_anomaly:
                if avg_anomaly_score < -0.5:
                    severity = "high"
                    anomaly_type = "highly_unusual_pattern"
                elif avg_anomaly_score < -0.2:
                    severity = "medium"
                    anomaly_type = "unusual_pattern"
                else:
                    severity = "low"
                    anomaly_type = "minor_deviation"
                
                description = f"Detected {anomaly_type} in crossing behavior with score {avg_anomaly_score:.3f}"
            else:
                severity = "low"
                anomaly_type = "normal"
                description = "Crossing patterns appear normal"
            
            return AnomalyDetection(
                is_anomaly=is_anomaly,
                anomaly_score=abs(avg_anomaly_score),
                anomaly_type=anomaly_type,
                description=description,
                severity=severity,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomalous behavior: {e}")
            raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")
    
    async def adapt_detection_sensitivity(self, camera_id: str, environmental_data: Dict, historical_performance: Dict) -> Dict:
        """Adapt virtual line sensitivity based on environmental conditions and historical data (Requirement 8.3)"""
        try:
            # Extract environmental factors
            lighting_level = environmental_data.get('lighting_level', 0.5)  # 0-1 scale
            weather_condition = environmental_data.get('weather', 'clear')
            temperature = environmental_data.get('temperature', 25.0)
            wind_speed = environmental_data.get('wind_speed', 0.0)
            visibility = environmental_data.get('visibility', 1.0)  # 0-1 scale
            
            # Extract historical performance metrics
            false_positive_rate = historical_performance.get('false_positive_rate', 0.01)
            detection_accuracy = historical_performance.get('detection_accuracy', 0.95)
            missed_detections = historical_performance.get('missed_detections', 0.05)
            
            # Base sensitivity (default)
            base_sensitivity = 0.7
            
            # Environmental adjustments
            lighting_adjustment = 0.0
            if lighting_level < 0.3:  # Low light conditions
                lighting_adjustment = 0.15  # Increase sensitivity
            elif lighting_level > 0.8:  # Bright conditions
                lighting_adjustment = -0.05  # Slightly decrease sensitivity
            
            weather_adjustment = 0.0
            weather_adjustments = {
                'rain': 0.1,
                'fog': 0.2,
                'snow': 0.15,
                'dust_storm': 0.25,
                'clear': 0.0
            }
            weather_adjustment = weather_adjustments.get(weather_condition, 0.0)
            
            # Temperature adjustment (extreme temperatures affect camera performance)
            temp_adjustment = 0.0
            if temperature < 0 or temperature > 45:
                temp_adjustment = 0.1
            
            # Visibility adjustment
            visibility_adjustment = (1.0 - visibility) * 0.2
            
            # Historical performance adjustments
            performance_adjustment = 0.0
            if false_positive_rate > 0.02:  # Too many false positives
                performance_adjustment = -0.1  # Decrease sensitivity
            elif missed_detections > 0.1:  # Too many missed detections
                performance_adjustment = 0.15  # Increase sensitivity
            
            # Calculate final sensitivity
            adjusted_sensitivity = base_sensitivity + lighting_adjustment + weather_adjustment + temp_adjustment + visibility_adjustment + performance_adjustment
            
            # Clamp sensitivity between 0.1 and 1.0
            adjusted_sensitivity = max(0.1, min(1.0, adjusted_sensitivity))
            
            # Calculate confidence threshold (inverse relationship with sensitivity)
            confidence_threshold = 1.0 - (adjusted_sensitivity * 0.5)  # Range: 0.5 to 0.95
            
            adaptation_result = {
                'camera_id': camera_id,
                'original_sensitivity': base_sensitivity,
                'adjusted_sensitivity': adjusted_sensitivity,
                'confidence_threshold': confidence_threshold,
                'adjustments': {
                    'lighting': lighting_adjustment,
                    'weather': weather_adjustment,
                    'temperature': temp_adjustment,
                    'visibility': visibility_adjustment,
                    'performance': performance_adjustment
                },
                'environmental_factors': environmental_data,
                'performance_metrics': historical_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Adapted detection sensitivity for camera {camera_id}: {adjusted_sensitivity:.3f}")
            return adaptation_result
            
        except Exception as e:
            logger.error(f"Error adapting detection sensitivity: {e}")
            raise HTTPException(status_code=500, detail=f"Sensitivity adaptation failed: {str(e)}")
    
    async def predict_hardware_maintenance(self, hardware_metrics: Dict) -> PredictionResult:
        """Predict hardware maintenance needs based on system health metrics"""
        try:
            # Extract hardware health metrics
            cpu_usage = hardware_metrics.get('cpu_usage', 0.0)
            memory_usage = hardware_metrics.get('memory_usage', 0.0)
            disk_usage = hardware_metrics.get('disk_usage', 0.0)
            temperature = hardware_metrics.get('temperature', 25.0)
            uptime_hours = hardware_metrics.get('uptime_hours', 0.0)
            error_rate = hardware_metrics.get('error_rate', 0.0)
            network_latency = hardware_metrics.get('network_latency', 0.0)
            
            # Create feature vector
            features = np.array([[
                cpu_usage, memory_usage, disk_usage, temperature,
                uptime_hours, error_rate, network_latency
            ]])
            
            # Normalize features
            if not hasattr(self.scalers['hardware_metrics'], 'mean_'):
                # Initialize with reasonable defaults if not fitted
                self.scalers['hardware_metrics'].fit(features)
            
            normalized_features = self.scalers['hardware_metrics'].transform(features)
            
            # Predict maintenance score (0-1, higher means more urgent)
            # Simple rule-based prediction (in production, use trained ML model)
            maintenance_score = 0.0
            
            # CPU usage factor
            if cpu_usage > 0.8:
                maintenance_score += 0.3
            elif cpu_usage > 0.6:
                maintenance_score += 0.1
            
            # Memory usage factor
            if memory_usage > 0.9:
                maintenance_score += 0.3
            elif memory_usage > 0.7:
                maintenance_score += 0.1
            
            # Temperature factor
            if temperature > 70:
                maintenance_score += 0.4
            elif temperature > 60:
                maintenance_score += 0.2
            
            # Error rate factor
            if error_rate > 0.05:
                maintenance_score += 0.3
            elif error_rate > 0.01:
                maintenance_score += 0.1
            
            # Uptime factor (very high uptime might indicate need for restart)
            if uptime_hours > 720:  # 30 days
                maintenance_score += 0.1
            
            # Determine risk level and recommendations
            if maintenance_score > 0.7:
                risk_level = "high"
                recommendations = [
                    "Schedule immediate maintenance",
                    "Check cooling system",
                    "Monitor system closely",
                    "Consider hardware replacement"
                ]
            elif maintenance_score > 0.4:
                risk_level = "medium"
                recommendations = [
                    "Schedule maintenance within 48 hours",
                    "Monitor temperature and performance",
                    "Check for software updates"
                ]
            elif maintenance_score > 0.2:
                risk_level = "low"
                recommendations = [
                    "Schedule routine maintenance",
                    "Continue monitoring"
                ]
            else:
                risk_level = "minimal"
                recommendations = ["System operating normally"]
            
            return PredictionResult(
                prediction_type="hardware_maintenance",
                confidence=0.85,  # Static confidence for rule-based system
                predicted_value=maintenance_score,
                risk_level=risk_level,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting hardware maintenance: {e}")
            raise HTTPException(status_code=500, detail=f"Hardware prediction failed: {str(e)}")
    
    async def _get_historical_crossings(self, camera_id: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve historical crossing data (mock implementation)"""
        # In production, this would query the actual database
        # For now, generate mock data for testing
        import random
        
        data = []
        current_date = start_date
        while current_date < end_date:
            # Simulate varying crossing patterns
            hour = current_date.hour
            # More crossings during dawn/dusk hours
            if 5 <= hour <= 8 or 17 <= hour <= 20:
                num_crossings = random.randint(0, 3)
            else:
                num_crossings = random.randint(0, 1)
            
            for _ in range(num_crossings):
                data.append({
                    'timestamp': current_date + timedelta(minutes=random.randint(0, 59)),
                    'camera_id': camera_id,
                    'confidence': random.uniform(0.7, 0.95),
                    'group_size': random.randint(1, 3),
                    'speed': random.uniform(0.5, 2.0),
                    'direction_consistency': random.uniform(0.8, 1.0)
                })
            
            current_date += timedelta(hours=1)
        
        return pd.DataFrame(data)

# Initialize the analytics engine
analytics_engine = PredictiveAnalyticsEngine()

# API Models
class HistoricalAnalysisRequest(BaseModel):
    camera_id: str
    days_back: int = 30

class AnomalyDetectionRequest(BaseModel):
    recent_crossings: List[Dict]

class SensitivityAdaptationRequest(BaseModel):
    camera_id: str
    environmental_data: Dict
    historical_performance: Dict

class MaintenancePredictionRequest(BaseModel):
    hardware_metrics: Dict

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "predictive-analytics", "timestamp": datetime.now()}

@app.post("/analyze/patterns")
async def analyze_patterns(request: HistoricalAnalysisRequest):
    """Analyze historical crossing patterns (Requirement 8.1)"""
    try:
        result = await analytics_engine.analyze_historical_patterns(
            request.camera_id, request.days_back
        )
        return asdict(result)
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/anomalies")
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalous crossing behavior (Requirement 8.1)"""
    try:
        result = await analytics_engine.detect_anomalous_behavior(request.recent_crossings)
        return asdict(result)
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adapt/sensitivity")
async def adapt_sensitivity(request: SensitivityAdaptationRequest):
    """Adapt detection sensitivity based on environmental conditions (Requirement 8.3)"""
    try:
        result = await analytics_engine.adapt_detection_sensitivity(
            request.camera_id,
            request.environmental_data,
            request.historical_performance
        )
        return result
    except Exception as e:
        logger.error(f"Sensitivity adaptation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/maintenance")
async def predict_maintenance(request: MaintenancePredictionRequest):
    """Predict hardware maintenance needs"""
    try:
        result = await analytics_engine.predict_hardware_maintenance(request.hardware_metrics)
        return asdict(result)
    except Exception as e:
        logger.error(f"Maintenance prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/status")
async def get_model_status():
    """Get status of predictive models"""
    return {
        "models": list(analytics_engine.models.keys()),
        "anomaly_detectors": list(analytics_engine.anomaly_detectors.keys()),
        "scalers": list(analytics_engine.scalers.keys()),
        "historical_data": {k: len(v) for k, v in analytics_engine.historical_data.items()},
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8007)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )