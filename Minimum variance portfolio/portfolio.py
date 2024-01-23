import numpy as np
import matplotlib.pyplot as plt

def obtener_portafolio_minima_varianza(retornos, covarianzas):
    inversa_covarianzas = np.linalg.inv(covarianzas)
    unos = np.ones(len(retornos))

    w_min_varianza = np.dot(inversa_covarianzas, unos) / np.dot(np.dot(unos, inversa_covarianzas), unos)
    w_min_varianza = w_min_varianza / np.sum(w_min_varianza)

    return w_min_varianza

def calcular_retorno_y_varianza(w, retornos, covarianzas):
    retorno = np.dot(w, retornos)
    varianza = np.dot(np.dot(w, covarianzas), w)
    return retorno, varianza

# Pedir al usuario la cantidad de activos y sus retornos
num_activos = int(input("Ingrese el número de activos: "))
retornos = np.array([float(input(f"Retorno del activo {i + 1}: ")) for i in range(num_activos)])

# Pedir al usuario la matriz de covarianzas
print("Ingrese la matriz de covarianzas:")
covarianzas = np.array([list(map(float, input().split())) for _ in range(num_activos)])

# Calcular el portafolio de mínima varianza
w_min_varianza = obtener_portafolio_minima_varianza(retornos, covarianzas)

# Imprimir resultados
print("\nPesos del portafolio de mínima varianza:")
for i in range(num_activos):
    print(f"Activo {i + 1}: {w_min_varianza[i]:.4f}")

# Calcular el retorno y la varianza del portafolio de mínima varianza
retorno_min_varianza, varianza_min_varianza = calcular_retorno_y_varianza(w_min_varianza, retornos, covarianzas)

print("\nRetorno del portafolio de mínima varianza:", retorno_min_varianza)
print("Varianza del portafolio de mínima varianza:", varianza_min_varianza)

# Generar la frontera de mínima varianza
pesos = []
varianzas = []

for tasa_retorno_objetivo in np.linspace(min(retornos), max(retornos), num=100):
    restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                     {'type': 'eq', 'fun': lambda w: np.dot(w, retornos) - tasa_retorno_objetivo}]

    resultado = minimize(lambda w: np.dot(np.dot(w, covarianzas), w), w_min_varianza, constraints=restricciones)
    peso_optimo = resultado.x

    pesos.append(peso_optimo)
    varianzas.append(np.dot(np.dot(peso_optimo, covarianzas), peso_optimo))

# Graficar la frontera de mínima varianza
plt.figure(figsize=(10, 6))
plt.scatter(varianzas, [r for r in np.linspace(min(retornos), max(retornos), num=100)], c=varianzas, cmap='viridis', marker='o')
plt.title('Frontera de Mínima Varianza')
plt.xlabel('Varianza')
plt.ylabel('Tasa de Retorno')
plt.colorbar(label='Varianza')
plt.show()
