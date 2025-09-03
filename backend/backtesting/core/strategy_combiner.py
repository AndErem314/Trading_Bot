"""
Strategy Combination Module

This module combines signals from multiple strategies trading the same pair
(e.g., multiple strategies all trading BTC/USDT).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """Methods for combining strategy signals"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    UNANIMOUS = "unanimous"
    ANY_SIGNAL = "any_signal"
    SCORE_BASED = "score_based"


@dataclass
class StrategyWeight:
    """Weight configuration for a strategy"""
    strategy_name: str
    weight: float = 1.0
    min_confidence: float = 0.0
    enabled: bool = True


class StrategyCombiner:
    """
    Combines signals from multiple strategies trading the same pair
    """
    
    def __init__(
        self,
        combination_method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE,
        min_strategies_agree: int = 2,
        signal_threshold: float = 0.5
    ):
        """
        Initialize the strategy combiner
        
        Args:
            combination_method: How to combine signals
            min_strategies_agree: Minimum strategies that must agree for majority vote
            signal_threshold: Threshold for generating final signal
        """
        self.combination_method = combination_method
        self.min_strategies_agree = min_strategies_agree
        self.signal_threshold = signal_threshold
        self.strategy_weights: Dict[str, StrategyWeight] = {}
        
    def add_strategy(self, strategy_name: str, weight: float = 1.0):
        """Add a strategy with its weight"""
        self.strategy_weights[strategy_name] = StrategyWeight(
            strategy_name=strategy_name,
            weight=weight,
            enabled=True
        )
        
    def combine_signals(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine signals from multiple strategies
        
        Args:
            strategy_signals: Dictionary mapping strategy names to signal DataFrames
            market_data: Market OHLCV data
            
        Returns:
            Combined signals DataFrame
        """
        if not strategy_signals:
            return pd.DataFrame()
            
        # Filter enabled strategies
        enabled_signals = {
            name: signals for name, signals in strategy_signals.items()
            if name in self.strategy_weights and self.strategy_weights[name].enabled
        }
        
        if not enabled_signals:
            logger.warning("No enabled strategies found")
            return pd.DataFrame()
            
        logger.info(f"Combining signals from {len(enabled_signals)} strategies using {self.combination_method.value}")
        
        # Align all signals to the same timestamps
        aligned_signals = self._align_signals(enabled_signals, market_data)
        
        # Combine based on method
        if self.combination_method == CombinationMethod.MAJORITY_VOTE:
            combined = self._majority_vote_combination(aligned_signals)
        elif self.combination_method == CombinationMethod.WEIGHTED_AVERAGE:
            combined = self._weighted_average_combination(aligned_signals)
        elif self.combination_method == CombinationMethod.UNANIMOUS:
            combined = self._unanimous_combination(aligned_signals)
        elif self.combination_method == CombinationMethod.ANY_SIGNAL:
            combined = self._any_signal_combination(aligned_signals)
        elif self.combination_method == CombinationMethod.SCORE_BASED:
            combined = self._score_based_combination(aligned_signals)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
            
        return combined
    
    def _align_signals(
        self,
        strategy_signals: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Align all strategy signals to the same timestamp index"""
        # Use market data timestamps as the reference
        base_index = market_data.index if isinstance(market_data.index, pd.DatetimeIndex) else pd.to_datetime(market_data['timestamp'])
        
        # Create a base DataFrame with market data timestamps
        aligned_df = pd.DataFrame(index=base_index)
        aligned_df['timestamp'] = base_index
        aligned_df['price'] = market_data['close'].values
        
        # Add each strategy's signals
        for strategy_name, signals in strategy_signals.items():
            # Ensure signals have timestamp index
            if 'timestamp' in signals.columns:
                signals = signals.set_index('timestamp')
            
            # Rename columns to include strategy name
            strategy_cols = {
                'signal': f'{strategy_name}_signal',
                'strength': f'{strategy_name}_strength'
            }
            
            # Reindex to match base timestamps and forward fill
            strategy_data = signals[['signal', 'strength']].reindex(base_index, method='ffill')
            strategy_data = strategy_data.rename(columns=strategy_cols)
            
            # Fill any remaining NaN values
            strategy_data.fillna(0, inplace=True)
            
            # Add to aligned DataFrame
            aligned_df = pd.concat([aligned_df, strategy_data], axis=1)
        
        return aligned_df
    
    def _majority_vote_combination(self, aligned_signals: pd.DataFrame) -> pd.DataFrame:
        """Combine signals using majority vote"""
        result = pd.DataFrame(index=aligned_signals.index)
        result['timestamp'] = aligned_signals['timestamp']
        result['price'] = aligned_signals['price']
        
        # Extract signal columns
        signal_cols = [col for col in aligned_signals.columns if col.endswith('_signal')]
        
        # Calculate votes for each row
        buy_votes = (aligned_signals[signal_cols] == 1).sum(axis=1)
        sell_votes = (aligned_signals[signal_cols] == -1).sum(axis=1)
        
        # Generate combined signal based on votes
        result['signal'] = 0
        result.loc[buy_votes >= self.min_strategies_agree, 'signal'] = 1
        result.loc[sell_votes >= self.min_strategies_agree, 'signal'] = -1
        
        # Calculate strength as percentage of strategies agreeing
        total_strategies = len(signal_cols)
        result['strength'] = 0.0
        result.loc[result['signal'] == 1, 'strength'] = buy_votes / total_strategies
        result.loc[result['signal'] == -1, 'strength'] = sell_votes / total_strategies
        
        # Add vote details
        result['buy_votes'] = buy_votes
        result['sell_votes'] = sell_votes
        result['total_strategies'] = total_strategies
        
        return result
    
    def _weighted_average_combination(self, aligned_signals: pd.DataFrame) -> pd.DataFrame:
        """Combine signals using weighted average"""
        result = pd.DataFrame(index=aligned_signals.index)
        result['timestamp'] = aligned_signals['timestamp']
        result['price'] = aligned_signals['price']
        
        # Calculate weighted signal score
        weighted_score = 0
        total_weight = 0
        
        for strategy_name, weight_config in self.strategy_weights.items():
            if not weight_config.enabled:
                continue
                
            signal_col = f'{strategy_name}_signal'
            strength_col = f'{strategy_name}_strength'
            
            if signal_col in aligned_signals.columns:
                # Weight both signal and strength
                weighted_signal = aligned_signals[signal_col] * aligned_signals[strength_col] * weight_config.weight
                weighted_score = weighted_score + weighted_signal
                total_weight += weight_config.weight
        
        # Normalize by total weight
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0
        
        # Generate final signal based on threshold
        result['signal'] = 0
        result.loc[normalized_score >= self.signal_threshold, 'signal'] = 1
        result.loc[normalized_score <= -self.signal_threshold, 'signal'] = -1
        
        # Strength is the absolute normalized score
        result['strength'] = np.abs(normalized_score).clip(0, 1)
        result['weighted_score'] = normalized_score
        
        return result
    
    def _unanimous_combination(self, aligned_signals: pd.DataFrame) -> pd.DataFrame:
        """Generate signal only when all strategies agree"""
        result = pd.DataFrame(index=aligned_signals.index)
        result['timestamp'] = aligned_signals['timestamp']
        result['price'] = aligned_signals['price']
        
        # Extract signal columns
        signal_cols = [col for col in aligned_signals.columns if col.endswith('_signal')]
        
        # Check unanimous agreement
        all_buy = (aligned_signals[signal_cols] == 1).all(axis=1)
        all_sell = (aligned_signals[signal_cols] == -1).all(axis=1)
        
        result['signal'] = 0
        result.loc[all_buy, 'signal'] = 1
        result.loc[all_sell, 'signal'] = -1
        
        # Strength is average of all strategy strengths when unanimous
        strength_cols = [col for col in aligned_signals.columns if col.endswith('_strength')]
        result['strength'] = 0.0
        result.loc[all_buy | all_sell, 'strength'] = aligned_signals.loc[all_buy | all_sell, strength_cols].mean(axis=1)
        
        return result
    
    def _any_signal_combination(self, aligned_signals: pd.DataFrame) -> pd.DataFrame:
        """Generate signal when any strategy signals"""
        result = pd.DataFrame(index=aligned_signals.index)
        result['timestamp'] = aligned_signals['timestamp']
        result['price'] = aligned_signals['price']
        
        # Extract signal columns
        signal_cols = [col for col in aligned_signals.columns if col.endswith('_signal')]
        
        # Check if any strategy signals
        any_buy = (aligned_signals[signal_cols] == 1).any(axis=1)
        any_sell = (aligned_signals[signal_cols] == -1).any(axis=1)
        
        # Handle conflicts (both buy and sell signals)
        result['signal'] = 0
        result.loc[any_buy & ~any_sell, 'signal'] = 1
        result.loc[any_sell & ~any_buy, 'signal'] = -1
        
        # For conflicts, use weighted average approach
        conflicts = any_buy & any_sell
        if conflicts.any():
            weighted_scores = self._calculate_weighted_scores(aligned_signals.loc[conflicts])
            result.loc[conflicts & (weighted_scores > 0), 'signal'] = 1
            result.loc[conflicts & (weighted_scores < 0), 'signal'] = -1
        
        # Strength is maximum strength among signaling strategies
        strength_cols = [col for col in aligned_signals.columns if col.endswith('_strength')]
        result['strength'] = 0.0
        
        for i in result.index:
            if result.loc[i, 'signal'] != 0:
                # Get strengths of strategies that agree with final signal
                agreeing_strengths = []
                for strategy_name in self.strategy_weights:
                    signal_col = f'{strategy_name}_signal'
                    strength_col = f'{strategy_name}_strength'
                    if signal_col in aligned_signals.columns:
                        if aligned_signals.loc[i, signal_col] == result.loc[i, 'signal']:
                            agreeing_strengths.append(aligned_signals.loc[i, strength_col])
                
                if agreeing_strengths:
                    result.loc[i, 'strength'] = max(agreeing_strengths)
        
        return result
    
    def _score_based_combination(self, aligned_signals: pd.DataFrame) -> pd.DataFrame:
        """Combine signals using a scoring system"""
        result = pd.DataFrame(index=aligned_signals.index)
        result['timestamp'] = aligned_signals['timestamp']
        result['price'] = aligned_signals['price']
        
        # Calculate composite score
        composite_score = pd.Series(0.0, index=aligned_signals.index)
        
        for strategy_name, weight_config in self.strategy_weights.items():
            if not weight_config.enabled:
                continue
                
            signal_col = f'{strategy_name}_signal'
            strength_col = f'{strategy_name}_strength'
            
            if signal_col in aligned_signals.columns:
                # Score = signal * strength * weight
                strategy_score = (aligned_signals[signal_col] * 
                                aligned_signals[strength_col] * 
                                weight_config.weight)
                composite_score += strategy_score
        
        # Normalize score to [-1, 1]
        max_possible_score = sum(w.weight for w in self.strategy_weights.values())
        if max_possible_score > 0:
            normalized_score = composite_score / max_possible_score
        else:
            normalized_score = composite_score
        
        # Generate signals based on score thresholds
        result['signal'] = 0
        result.loc[normalized_score >= self.signal_threshold, 'signal'] = 1
        result.loc[normalized_score <= -self.signal_threshold, 'signal'] = -1
        
        result['strength'] = np.abs(normalized_score).clip(0, 1)
        result['composite_score'] = normalized_score
        
        return result
    
    def _calculate_weighted_scores(self, signals_subset: pd.DataFrame) -> pd.Series:
        """Calculate weighted scores for a subset of signals"""
        weighted_score = pd.Series(0.0, index=signals_subset.index)
        
        for strategy_name, weight_config in self.strategy_weights.items():
            signal_col = f'{strategy_name}_signal'
            strength_col = f'{strategy_name}_strength'
            
            if signal_col in signals_subset.columns:
                weighted_score += (signals_subset[signal_col] * 
                                 signals_subset[strength_col] * 
                                 weight_config.weight)
        
        return weighted_score
    
    def analyze_strategy_agreement(
        self,
        strategy_signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze how often strategies agree with each other"""
        if len(strategy_signals) < 2:
            return {}
        
        aligned_signals = self._align_signals(strategy_signals, pd.DataFrame(index=strategy_signals[list(strategy_signals.keys())[0]].index))
        
        # Calculate pairwise agreement
        signal_cols = [col for col in aligned_signals.columns if col.endswith('_signal')]
        
        agreement_matrix = {}
        for i, col1 in enumerate(signal_cols):
            strategy1 = col1.replace('_signal', '')
            agreement_matrix[strategy1] = {}
            
            for j, col2 in enumerate(signal_cols):
                strategy2 = col2.replace('_signal', '')
                
                if i != j:
                    # Calculate percentage of time strategies agree
                    total_signals = ((aligned_signals[col1] != 0) | (aligned_signals[col2] != 0)).sum()
                    if total_signals > 0:
                        agreements = (aligned_signals[col1] == aligned_signals[col2]).sum()
                        agreement_pct = agreements / total_signals * 100
                    else:
                        agreement_pct = 0
                    
                    agreement_matrix[strategy1][strategy2] = agreement_pct
        
        # Calculate overall statistics
        all_agreements = []
        for s1 in agreement_matrix:
            for s2 in agreement_matrix[s1]:
                all_agreements.append(agreement_matrix[s1][s2])
        
        stats = {
            'agreement_matrix': agreement_matrix,
            'avg_agreement': np.mean(all_agreements) if all_agreements else 0,
            'min_agreement': np.min(all_agreements) if all_agreements else 0,
            'max_agreement': np.max(all_agreements) if all_agreements else 0
        }
        
        return stats
