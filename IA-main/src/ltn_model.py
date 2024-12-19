import torch
import ltn
import numpy as np

class TrainLTN:
    """
    Implementa Logic Tensor Networks para o problema do Trem de Michalski
    """
    def __init__(self):
        # Definir predicados
        self.predicates = {
            # Predicados principais
            'num_cars': ltn.Predicate.MLP([1, 10, 1]),
            'num_loads': ltn.Predicate.MLP([1, 10, 1]),
            'num_wheels': ltn.Predicate.MLP([1, 10, 1]),
            'length': ltn.Predicate.MLP([1, 10, 1]),
            'shape': ltn.Predicate.MLP([1, 10, 1]),
            'load_shape': ltn.Predicate.MLP([1, 10, 1]),

            # Predicados para adjacência
            'next_crc': ltn.Predicate.MLP([2, 10, 1]),
            'next_hex': ltn.Predicate.MLP([2, 10, 1]),
            'next_rec': ltn.Predicate.MLP([2, 10, 1]),
            'next_tri': ltn.Predicate.MLP([2, 10, 1]),

            # Predicado de direção
            'direction': ltn.Predicate.MLP([1, 2, 1])
        }

    def create_variables(self, data):
        """
        Cria variáveis LTN a partir dos dados
        """
        n_samples = len(data)
        self.variables = {
            't': ltn.Variable('train', torch.from_numpy(
                data.values
            ).float())
        }

    def create_axioms(self):
        """
        Cria axiomas baseados nas três teorias principais
        """
        # Teoria A: vagão curto e fechado -> leste
        axiom_a = ltn.Axiom(
            'short_closed_east',
            ltn.Forall(
                self.variables['t'],
                ltn.Implies(
                    ltn.And(
                        self.predicates['length'](self.variables['t']),
                        self.predicates['shape'](self.variables['t'])
                    ),
                    self.predicates['direction'](self.variables['t'])
                )
            )
        )

        # Teoria B: dois vagões ou teto irregular -> oeste
        axiom_b = ltn.Axiom(
            'two_cars_or_irregular',
            ltn.Forall(
                self.variables['t'],
                ltn.Implies(
                    ltn.Or(
                        self.predicates['num_cars'](self.variables['t']) == 2,
                        self.predicates['shape'](self.variables['t']) == 'irregular'
                    ),
                    ~self.predicates['direction'](self.variables['t'])
                )
            )
        )

        # Teoria C: múltiplos tipos de carga -> leste
        axiom_c = ltn.Axiom(
            'multiple_loads',
            ltn.Forall(
                self.variables['t'],
                ltn.Implies(
                    self.predicates['num_loads'](self.variables['t']) > 2,
                    self.predicates['direction'](self.variables['t'])
                )
            )
        )

        return [axiom_a, axiom_b, axiom_c]

    def train(self, data, epochs=100, learning_rate=0.01):
        """
        Treina o modelo LTN
        """
        self.create_variables(data)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Calcular satisfação dos axiomas
            sat = torch.mean(torch.stack([
                axiom(data) for axiom in self.create_axioms()
            ]))

            loss = 1 - sat
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        return losses

    def predict_direction(self, train_data):
        """
        Prediz direção de um trem
        """
        with torch.no_grad():
            direction = self.predicates['direction'](
                torch.from_numpy(train_data.values).float()
            )
        return 'east' if direction > 0.5 else 'west'
