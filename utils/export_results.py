import pandas as pd

def export_results(prices_history, quantities_history, monopoly_history, 
                   nash_history, rewards_history, metric_history, 
                   pi_N_history, pi_M_history,
                   costs_history, exp_name):
    
    exp_dict = {'prices_history': prices_history, 'quantities_history': quantities_history,
                'rewards_history': rewards_history, 'costs_history': costs_history,
                'monopoly_history': monopoly_history, 'nash_history': nash_history, 
                'nash_utilities': pi_N_history, 'monopoly_utilities': pi_M_history,
                'metric_history': metric_history}
    
    #print('prices_history:', len(prices_history))
    #print('quantities_history:', len(quantities_history))
    #print('rewards_history:', len(rewards_history))
    #print('costs_history:', len(costs_history))
    #print('monopoly_history:', len(monopoly_history))
    #print('nash_history:', len(nash_history))
    #print('metric_history:', len(metric_history))
    
    exp_metrics = pd.DataFrame(exp_dict)
    
    #exp_metrics['nash_utilities'] = (exp_metrics['nash_history'] - exp_metrics['costs_history']) * exp_metrics['quantities_history']
    
    exp_metrics.to_csv(f'metrics/{exp_name}.csv', sep = '\t')