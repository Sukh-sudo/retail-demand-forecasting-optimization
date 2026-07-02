import pandas as pd
import pulp


def run_inventory_optimization(
    input_data: pd.DataFrame,
    scenario_name: str = "Base Case",
    demand_multiplier: float = 1.0,
    capacity_rate: float = 0.95,
    holding_cost_rate: float = 0.15,
    stockout_cost_multiplier: float = 2.0,
    safety_stock_rate: float = 0.20,
):
    """
    Run inventory allocation optimization for a given scenario.
    """

    data = input_data.copy()

    required_columns = [
        "item_id",
        "store_id",
        "state_id",
        "forecast_demand",
        "actual_demand",
        "average_price",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Input data is missing required columns: {missing_columns}"
        )

    data["scenario_forecast_demand"] = (
        data["forecast_demand"] * demand_multiplier
    )

    data["holding_cost"] = (
        data["average_price"] * holding_cost_rate
    )

    data["stockout_cost"] = (
        data["average_price"] * stockout_cost_multiplier
    )

    data["safety_stock"] = (
        data["scenario_forecast_demand"] * safety_stock_rate
    )

    data["target_inventory"] = (
        data["scenario_forecast_demand"]
        + data["safety_stock"]
    )

    warehouse_capacity = (
        data["target_inventory"].sum() * capacity_rate
    )

    model = pulp.LpProblem(
        f"Inventory_Optimization_{scenario_name}",
        pulp.LpMinimize
    )

    items = data.index.tolist()

    inventory = pulp.LpVariable.dicts(
        "inventory",
        items,
        lowBound=0,
        cat="Continuous"
    )

    shortage = pulp.LpVariable.dicts(
        "shortage",
        items,
        lowBound=0,
        cat="Continuous"
    )

    excess = pulp.LpVariable.dicts(
        "excess",
        items,
        lowBound=0,
        cat="Continuous"
    )

    model += pulp.lpSum(
        data.loc[i, "holding_cost"] * excess[i]
        + data.loc[i, "stockout_cost"] * shortage[i]
        for i in items
    )

    for i in items:
        target = data.loc[i, "target_inventory"]

        model += inventory[i] + shortage[i] >= target
        model += excess[i] >= inventory[i] - target

    model += pulp.lpSum(
        inventory[i] for i in items
    ) <= warehouse_capacity

    model.solve(
        pulp.PULP_CBC_CMD(msg=False)
    )

    data["recommended_inventory"] = [
        inventory[i].value()
        for i in items
    ]

    data["expected_shortage"] = [
        shortage[i].value()
        for i in items
    ]

    data["expected_excess"] = [
        excess[i].value()
        for i in items
    ]

    data["actual_shortage"] = (
        data["actual_demand"]
        - data["recommended_inventory"]
    ).clip(lower=0)

    data["actual_excess"] = (
        data["recommended_inventory"]
        - data["actual_demand"]
    ).clip(lower=0)

    data["scenario"] = scenario_name
    data["warehouse_capacity"] = warehouse_capacity
    data["solver_status"] = pulp.LpStatus[model.status]
    data["objective_value"] = pulp.value(model.objective)

    total_actual_demand = data["actual_demand"].sum()
    total_actual_shortage = data["actual_shortage"].sum()
    total_recommended_inventory = data["recommended_inventory"].sum()

    service_level = (
        1 - total_actual_shortage / total_actual_demand
        if total_actual_demand > 0
        else 0
    )

    capacity_used = (
        total_recommended_inventory / warehouse_capacity
        if warehouse_capacity > 0
        else 0
    )

    summary = pd.DataFrame({
        "scenario": [scenario_name],
        "solver_status": [pulp.LpStatus[model.status]],
        "objective_value": [pulp.value(model.objective)],
        "total_forecast_demand": [data["scenario_forecast_demand"].sum()],
        "total_target_inventory": [data["target_inventory"].sum()],
        "warehouse_capacity": [warehouse_capacity],
        "total_recommended_inventory": [total_recommended_inventory],
        "total_actual_demand": [total_actual_demand],
        "total_actual_shortage": [total_actual_shortage],
        "total_actual_excess": [data["actual_excess"].sum()],
        "service_level": [service_level],
        "capacity_used": [capacity_used],
        "demand_multiplier": [demand_multiplier],
        "capacity_rate": [capacity_rate],
        "holding_cost_rate": [holding_cost_rate],
        "stockout_cost_multiplier": [stockout_cost_multiplier],
        "safety_stock_rate": [safety_stock_rate],
    })

    return data, summary