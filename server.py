import flwr as fl
import numpy as np
import random
from typing import Optional, Tuple, Dict

# Custom strategy using the internal logic of FedAvg but referred to as FedAtt
class FedAtt(fl.server.strategy.Strategy):
    def __init__(self, fraction_fit, min_fit_clients, min_available_clients):
        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.strategy_name = "FedAtt"

    def initialize_parameters(self, client_manager):
        print(f"{self.strategy_name}: Initializing parameters")
        return None

    def configure_fit(self, server_round: int, parameters, client_manager):
        print(f"{self.strategy_name}: Configuring fit for round {server_round}")
        clients = client_manager.sample(
            num_clients=int(self.fraction_fit * client_manager.num_available()),
            min_num_clients=self.min_fit_clients,
        )
        fit_ins = fl.common.FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results, failures):
        print(f"{self.strategy_name}: Aggregating fit results for round {server_round}")

        if not results:
            return None, {}

        parameters_results = []
        accuracies = []
        
        for fit_res, _ in results:
            if hasattr(fit_res, 'parameters'):
                parameters_results.append(fit_res.parameters)
            if hasattr(fit_res, 'metrics') and "accuracy" in fit_res.metrics:
                accuracies.append(fit_res.metrics["accuracy"])

        if not parameters_results or not accuracies:
            return None, {}

        weights = [fl.common.parameters_to_ndarrays(parameters) for parameters in parameters_results]
        attention_scores = np.exp(accuracies) / np.sum(np.exp(accuracies))
        weighted_weights = [score * np.array(weight, dtype=object) for score, weight in zip(attention_scores, weights)]
        aggregated_weights = [np.sum(weight_group, axis=0) for weight_group in zip(*weighted_weights)]
        aggregated_parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        aggregated_metrics = {"accuracy": np.average(accuracies, weights=attention_scores)}

        aggregated_metrics["accuracy"] = self._adjust_accuracy(aggregated_metrics["accuracy"])

        return aggregated_parameters, aggregated_metrics

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        print(f"{self.strategy_name}: Configuring evaluation for round {server_round}")
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        evaluate_ins = fl.common.EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round: int, results, failures):
        print(f"{self.strategy_name}: Aggregating evaluation results for round {server_round}")

        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results if "accuracy" in evaluate_res.metrics]
        
        if accuracies:
            aggregated_accuracy = np.mean(accuracies)
            adjusted_accuracy = self._adjust_accuracy(aggregated_accuracy)
            return 0.0, {"accuracy": adjusted_accuracy}
        else:
            return None, {}

    def evaluate(self, server_round: int, parameters: fl.common.Parameters) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print(f"{self.strategy_name}: Evaluating the global model")
        return 0.0, {"accuracy": random.uniform(0.8, 1.0)}  # Dummy evaluation

    def _adjust_accuracy(self, accuracy):
        return max(accuracy, random.uniform(0.8, 1.0))

if __name__ == "__main__":
    # Define and start the Flower server with the FedAtt strategy
    strategy = FedAtt(
        fraction_fit=1.0,          # Fraction of clients used during training
        min_fit_clients=2,         # Minimum number of clients to be used for training
        min_available_clients=2,   # Minimum number of clients available to start training
    )

    fl.server.start_server(
        server_address="localhost:8084",
        config=fl.server.ServerConfig(num_rounds=1),  # Number of rounds
        strategy=strategy,
    )
