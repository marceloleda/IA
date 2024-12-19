import pandas as pd

class TheoryVerifier:
    def __init__(self, model):
        self.model = model

    def verify_all_theories(self, data):
        results = {
            'theory_a': self.verify_theory_a(data),
            'theory_b': self.verify_theory_b(data),
            'theory_c': self.verify_theory_c(data)
        }
        return results

    def verify_theory_a(self, data):
        """Vagão curto e fechado -> leste"""
        correct = 0
        total = len(data)

        for _, train in data.iterrows():
            # Verifica se tem vagão curto e fechado
            has_short_closed = any(
                train[f'length{i}'] == 'short' and
                train[f'shape{i}'] == 'closed'
                for i in range(1, 5)
            )

            # Verifica se vai para leste
            goes_east = train['Class_attribute'] == 'east'

            if has_short_closed == goes_east:
                correct += 1

        return correct / total

    def verify_theory_b(self, data):
        """Dois vagões ou teto irregular -> oeste"""
        correct = 0
        total = len(data)

        for _, train in data.iterrows():
            # Verifica condições
            has_two_cars = train['Number_of_cars'] == 2
            has_irregular = any(
                'irregular' in str(train[f'shape{i}'])
                for i in range(1, 5)
            )

            # Verifica direção
            goes_west = train['Class_attribute'] == 'west'

            if (has_two_cars or has_irregular) == goes_west:
                correct += 1

        return correct / total

    def verify_theory_c(self, data):
        """Múltiplos tipos de carga -> leste"""
        correct = 0
        total = len(data)

        for _, train in data.iterrows():
            # Conta tipos diferentes de carga
            load_types = set(
                train[f'load_shape{i}']
                for i in range(1, 5)
                if pd.notna(train[f'load_shape{i}'])
            )

            has_multiple_loads = len(load_types) > 2
            goes_east = train['Class_attribute'] == 'east'

            if has_multiple_loads == goes_east:
                correct += 1

        return correct / total
