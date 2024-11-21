import pandas as pd

def parse_dependencias(funcionales):
    """
    Convierte una lista de dependencias funcionales en texto al formato esperado.
    Ejemplo: "#P, #H ⟶ #P, #H" -> ([#P, #H], #P)
    """
    parsed = []
    for df in funcionales:
        # Separar determinantes y determinados por '⟶'
        izquierda, derecha = df.split('⟶')
        # Limpiar espacios y separar por comas
        izquierda = [x.strip() for x in izquierda.split(',')]
        derecha = [x.strip() for x in derecha.split(',')]
        parsed.append((izquierda, derecha[0]))  # Retornar como tuplas
    return parsed

def verificar_plj(U, F, descomposicion, filename):
    """
    Verifica si una descomposición cumple con la Propiedad de Joint sin Pérdidas (PLJ).
    
    Args:
        U (list): Atributos del esquema relacional.
        F (list): Dependencias funcionales, donde cada DF es una tupla (X, A), con X como lista de atributos.
        descomposicion (list): Lista de descomposiciones, cada una es un conjunto de atributos Ui.

    Returns:
        bool: True si la descomposición cumple con la PLJ, False en caso contrario.
    """
    # Paso 1: Construir la tabla inicial
    n = len(U)  # Número de columnas
    k = len(descomposicion)  # Número de filas

    nombres_descomposiciones = []
    for i, des in enumerate(descomposicion):
        attrs = ""
        for j, attr in enumerate(des):
            attrs += f"{attr}" + (", " if j < len(des) - 1 else "")
        nombres_descomposiciones.append( f"R{i+1}({attrs})")
        
    # Crear una tabla de k filas y n columnas
    tabla = pd.DataFrame([[f"b{i+1}{j+1}" for j in range(n)] for i in range(k)], columns=U)
    tabla.index = nombres_descomposiciones  # Usar nombres de las descomposiciones como índice

    # Asignar valores "aj" para atributos que están en la descomposición
    for i, Ui in enumerate(descomposicion):
        for atributo in Ui:
            if atributo in U:  # Solo si el atributo está en U
                tabla.loc[nombres_descomposiciones[i], atributo] = f"a{U.index(atributo) + 1}"

    # Mostrar la tabla inicial
    with open(filename, 'a') as file:
        file.write("Tabla inicial: \n")
        file.write(tabla.to_string()  + "\n\n")
    
        # Paso 2: Verificar dependencias funcionales y modificar la tabla
        cambios = True
        i = 1
        while cambios:
            cambios = False
            for X, A in F:
                # Encontrar todas las filas donde los valores en las columnas de X coinciden
                filas_coincidentes = tabla[tabla[X].duplicated(keep=False)]

                # Igualar los símbolos en la columna de A para estas filas
                if not filas_coincidentes.empty:
                    simbolos = filas_coincidentes[A].unique()
                    if len(simbolos) > 1:  # Si hay más de un símbolo
                        cambios = True
                        simbolo_final = next((s for s in simbolos if "a" in s), simbolos[0])
                        tabla.loc[filas_coincidentes.index, A] = simbolo_final

                        # Mostrar la tabla después de procesar las dependencias funcionales
                        file.write(f"Tabla después de la iteración {i} ({', '.join(X)} ⟶ {A}): \n")
                        file.write(tabla.to_string() + "\n\n")
                        i+=1

            # Verificar si alguna fila contiene solo símbolos "aj"
            if any(all("a" in tabla.loc[i, col] for col in tabla.columns) for i in tabla.index):
                return True

    # Si no se encontró una fila con todos los símbolos "aj", la descomposición no cumple la PLJ
    return False
