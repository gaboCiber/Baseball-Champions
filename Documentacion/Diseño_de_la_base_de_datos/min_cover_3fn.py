from itertools import chain, combinations  # Para manejar conjuntos y generar combinaciones
import pandas as pd  # Para manejar estructuras tabulares (DataFrames)
import plj  # Módulo externo, probablemente implementa la verificación de PLJ (Propiedad de Join sin Pérdidas)

redundant_attrs = []
redundant_dependencies = []

# Genera el conjunto potencia de un conjunto `s`
def power_set(s):
    """
    Args:
        s (list/set): Un conjunto de elementos.

    Returns:
        list: Una lista de subconjuntos (conjunto potencia), excluyendo el conjunto vacío.
    """
    # Genera todas las combinaciones posibles para tamaños de 1 hasta len(s)
    power_set = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))
    # Convierte las combinaciones (tuplas) en conjuntos
    return [set(subset) for subset in power_set]

# Genera el primer cubrimiento (First Cover) a partir de las dependencias funcionales
def first_cover(DFs):
    """
    Args:
        DFs (list): Lista de dependencias funcionales como strings (formato: 'A ⟶ B').

    Returns:
        tuple: El primer cubrimiento como lista de tuplas (determinantes, determinados) y el universo de atributos.
    """
    fc = []  # Lista para almacenar las dependencias funcionales procesadas
    uni = set()  # Conjunto de todos los atributos únicos (universo)
    
    for df in DFs:
        # Divide cada dependencia funcional en lado izquierdo (X) y lado derecho (A)
        left, right = df.split('⟶')
        
        left_splited = []  # Lista para almacenar los atributos del lado izquierdo
        for l in left.split(','):  # Divide los determinantes por comas
            l = l.strip()  # Elimina espacios en blanco
            left_splited.append(l)  # Agrega a la lista
            uni.add(l)  # Agrega al universo
        
        for r in right.split(','):  # Divide los determinados por comas
            r = r.strip()  # Elimina espacios en blanco
            fc.append((left_splited, [r]))  # Agrega la dependencia como una tupla
            uni.add(r)  # Agrega al universo

    return fc, uni  # Devuelve el primer cubrimiento y el universo

# Genera el segundo cubrimiento eliminando atributos redundantes
def second_cover(fc):
    """
    Args:
        fc (list): Lista de dependencias funcionales (formato: [(X, A)]).

    Returns:
        list: Segundo cubrimiento con dependencias funcionales minimizadas.
    """
    sc = []  # Lista para almacenar el segundo cubrimiento

    print("\nAtributos redundantes")  # Mostrar redundancias
    for l, r in fc:  # Iterar sobre todas las dependencias funcionales
        add = True  # Bandera para decidir si agregar la dependencia
        
        # Si hay más de un determinante, buscamos posibles redundancias
        if len(l) > 1:
            ps = power_set(l)  # Genera el conjunto potencia de los determinantes
            ps.pop(len(ps) - 1)  # Elimina el conjunto completo (original)
            
            # Verifica si el atributo determinado ya está cubierto por un subconjunto
            if set(r) in ps:
                add = False
            else:
                # Recorre las dependencias para buscar redundancias
                for l2, r2 in fc:
                    if (l2 in ps and r == r2) or (r2 in ps):
                        add = False
                        break

        if add:  # Si no es redundante
            sc.append((l, r))  # Agrega la dependencia funcional al segundo cubrimiento
        else:
            redundant_attrs.append((l,r))
            print(f"{l} ⟶ {r}")  # Muestra los atributos redundantes

    return sc  # Devuelve el segundo cubrimiento

# Realiza inferencia hacia atrás para deducir los determinantes de un conjunto
def backwards_inference(l, source, history):
    """
    Args:
        l (list): Conjunto de atributos iniciales.
        source (set): Atributos fuente conocidos.
        history (dict): Historial de inferencias.

    Returns:
        set: Conjunto deducido de determinantes.
    """
    trueL = set(l.copy())  # Copia inicial del conjunto de atributos
    change = True  # Bandera para indicar cambios
    
    while change:
        change = False
        for simple in trueL:
            # Si el atributo no está en la fuente, intentamos deducirlo
            if simple not in source:
                trueL.remove(simple)  # Elimina el atributo no deducible
                for i in history[frozenset([simple])]:  # Busca inferencias previas
                    trueL.add(i)  # Agrega los atributos inferidos
                change = True
                break

    return trueL  # Devuelve los determinantes deducidos

# Verifica si un atributo está determinado en las dependencias funcionales
def check(r, sc):
    """
    Args:
        r (str): Atributo a verificar.
        sc (list): Lista de dependencias funcionales.

    Returns:
        bool: True si el atributo está determinado, False en caso contrario.
    """
    for i, d in sc:
        if r == d:
            return True  # Retorna True si el atributo está determinado
    return False

# Identifica los atributos que no son determinantes
def get_non_determinant(sc, uni):
    """
    Args:
        sc (list): Lista de dependencias funcionales.
        uni (set): Universo de atributos.

    Returns:
        list: Lista de atributos que no son determinantes.
    """
    non = [i for i in uni]  # Inicializa con todos los atributos del universo
    for left, right in sc:
        for l in left:
            if l in non:
                non.remove(l)  # Elimina los determinantes del universo
    return non  # Devuelve los atributos no determinantes

# Calcula el cierre desde un conjunto fuente hasta un atributo objetivo
def closure_of_source_to_target(source, target, sc, non_det):
    """
    Calcula el cierre desde un conjunto fuente hasta un atributo objetivo, verificando las dependencias.

    Args:
        source (list): Conjunto de atributos iniciales.
        target (str): Atributo objetivo.
        sc (list): Lista de dependencias funcionales (formato: [(determinantes, determinados)]).
        non_det (list): Lista de atributos no determinantes.

    Returns:
        list/bool: El conjunto alcanzado si es posible llegar al objetivo, False en caso contrario.
    """
    ps = power_set(source)  # Genera el conjunto potencia de los atributos fuente
    change = True  # Bandera para indicar cambios
    history = {}  # Historial para rastrear inferencias

    while change:
        change = False
        for l, r in sc:
            # Si el conjunto de determinantes está en el conjunto potencia
            if set(l) in ps:
                # Si alcanzamos el atributo objetivo
                if r == target:
                    return backwards_inference(l, source, history)

                # Verifica si el atributo determinado es válido para agregar
                is_right_a_determinat = r[0] not in non_det
                is_right_in_ps = set(r) not in ps
                if is_right_a_determinat and is_right_in_ps:
                    # Actualiza el conjunto potencia con el atributo determinado
                    lenght = len(ps)
                    for s in ps:
                        ps.append(s.union(set(r)))
                        lenght -= 1
                        if(lenght == 0):
                            break
                    ps.append(set(r))
                    history[frozenset(r)] = l  # Registra la inferencia
                    change = True
                    break

    return False  # Devuelve False si no se pudo alcanzar el objetivo


# Calcula el tercer cubrimiento (Third Cover)
def third_cover(sc, uni):
    """
    Args:
        sc (list): Segundo cubrimiento (dependencias funcionales).
        uni (set): Universo de atributos.

    Returns:
        list: Tercer cubrimiento minimizado.
    """
    tc = []  # Lista para el tercer cubrimiento
    i = 0
    non_det = get_non_determinant(sc, uni)  # Obtiene los atributos no determinantes
    
    print("\nDependencias redundantes")  # Mostrar dependencias redundantes
    while i < len(sc):
        l, r = sc[i]
        scTemp = sc.copy()  # Copia temporal del segundo cubrimiento
        scTemp.pop(i)  # Elimina la dependencia actual para probar redundancias
        
        if l != r:  # Verifica que no sea trivial
            # Calcula el cierre y verifica redundancias
            L = closure_of_source_to_target(l, r, scTemp, non_det) if check(r, scTemp) else None
            if L:
                if (list(L), r) not in scTemp and set(L) != set(l):
                    tc.append((L, r))  # Agrega la dependencia minimizada
                    scTemp.insert(i, (L, r))  # Inserta la dependencia en la copia
                    print(f"{l} ⟶ {r} === {L} ⟶ {r}")
                else:
                    i -= 1  # Retrocede para revalidar
                    redundant_dependencies.append((l,r))
                    print(f"{l} ⟶ {r}")
                sc = scTemp
            else:
                tc.append((l, r))  # Agrega la dependencia original
        else:
            redundant_dependencies.append((l,r))
            print(f"{l} ⟶ {r} (iguales (0-0))")
        i += 1  # Avanza al siguiente índice

    return tc  # Devuelve el tercer cubrimiento

# Agrupa dependencias funcionales por sus determinantes
def group(tc, uni, DFs):
    """
    Agrupa las dependencias funcionales por sus determinantes y añade atributos triviales.

    Args:
        tc (list): Lista de dependencias funcionales (formato: [(X, A)]).
        uni (set): Conjunto de todos los atributos únicos (universo de atributos).

    Returns:
        dict: Diccionario con claves como conjuntos de determinantes (frozenset) y valores como listas de atributos determinados.
    """
    tcg = {}  # Diccionario para almacenar las agrupaciones de dependencias
    trivial = uni.copy()  # Copia del universo para procesar los atributos triviales

    # Iterar sobre todas las dependencias funcionales del tercer cubrimiento
    for l, r in tc:
        lset = frozenset(l)  # Convierte los determinantes en un conjunto inmutable
        if lset in tcg:
            # Si ya existe el grupo, agrega el atributo determinado
            tcg[lset].append(r[0])
        else:
            # Si no existe, crea un nuevo grupo con la dependencia
            tcg[lset] = r.copy()
        
        # Elimina los determinantes del conjunto de atributos triviales
        for i in l:
            if i in trivial:
                trivial.remove(i)
        # Elimina el atributo determinado si está en los triviales
        if r[0] in trivial:
            trivial.remove(r[0])

    # Agregar los atributos restantes (triviales) como dependencias de sí mismos
    for tri in trivial:
        tcg[frozenset([tri])] = [tri]
        
    for df in DFs:
        left , right = left, right = df.split('⟶')
        left = set([i.strip() for i in left.split(',')])
        right = set([i.strip() for i in right.split(',')])
        
        change = False
        for des in tcg:
            l = closure_of_source_to_universe(des,tc)
            r = closure_of_source_to_universe(tcg[des], tc)
            if l.issuperset(left) and r.issuperset(right):
                change = True
                break
        
        if not change and not (list(left), list(right)) in redundant_dependencies :
            tcg[frozenset(left)] = list(right)
            
        
    return tcg  # Devuelve el diccionario agrupado


# Genera las descomposiciones en Tercera Forma Normal (3FN)
def third_normal_form(tcgr):
    """
    Genera la descomposición en Tercera Forma Normal (3FN) a partir de las agrupaciones.

    Args:
        tcgr (dict): Diccionario de agrupaciones de dependencias funcionales.

    Returns:
        dict: Diccionario con las descomposiciones, donde cada entrada es:
              {i: (U, [(X, A)])}, siendo U el conjunto de atributos de la relación y [(X, A)] las dependencias.
    """
    descompositions = {}  # Diccionario para almacenar las descomposiciones

    # Iterar sobre las agrupaciones
    for left in tcgr:
        F = (list(left), tcgr[left])  # Dependencia funcional actual como una tupla (X, A)
        U = set(left).union(set(tcgr[left]))  # Unión de determinantes y determinados para formar la relación

        # Verificar si la relación ya existe o se debe crear una nueva
        for i in range(len(descompositions) + 1):
            if i == len(descompositions):
                # Crear una nueva relación si no existe
                descompositions[i] = (U, [F])
            elif descompositions[i][0].issuperset(U):
                # Si la relación existente contiene los mismos atributos, añade la dependencia
                descompositions[i][1].append(F)
                break

    return descompositions  # Devuelve el diccionario con las descomposiciones

# Genera la cobertura mínima y descomposición en 3FN
def min_cover_3fn(DFs):
    """
    Genera la cobertura mínima y la descomposición en 3FN (Tercera Forma Normal).

    Args:
        DFs (list): Lista de dependencias funcionales.

    Returns:
        tuple: Elementos de la cobertura mínima y la descomposición.
    """
    fc, uni = first_cover(DFs)  # Primer cubrimiento
    sc = second_cover(fc)  # Segundo cubrimiento
    tc = third_cover(sc, uni)  # Tercer cubrimiento
    tcgr = group(tc, uni, DFs)  # Agrupación de dependencias
    des = third_normal_form(tcgr)  # Descomposición en 3FN

    return uni, fc, sc, tc, tcgr, des  # Devuelve los elementos generados


# Calcula el cierre de un conjunto fuente hacia todo el universo de atributos
def closure_of_source_to_universe(source, tc):
    """
    Calcula el cierre de un conjunto de atributos hacia todo el universo.

    Args:
        source (list): Conjunto fuente inicial.
        tc (list): Lista de dependencias funcionales.

    Returns:
        set: Conjunto alcanzado (cierre completo).
    """
    x0 = set(source)  # Inicializa el conjunto alcanzado
    change = True  # Bandera para indicar cambios
    
    while change:
        change = False
        for left, right in tc:
            # Si el conjunto fuente contiene los determinantes y el determinado no está presente
            if x0.issuperset(set(left)) and right[0] not in x0:
                x0.add(right[0])  # Agrega el atributo alcanzado
                change = True  # Marca un cambio
    
    return x0  # Devuelve el conjunto alcanzado


# Calcula las claves candidatas
def candidate_keys(DFs):
    """
    Calcula todas las claves candidatas de un esquema relacional.

    Args:
        DFs (list): Lista de dependencias funcionales.

    Returns:
        list: Lista de claves candidatas.
    """
    fc, uni = first_cover(DFs)  # Primer cubrimiento
    reacheable_attrs = set(r[0] for l, r in fc)  # Atributos que se pueden alcanzar
    non_reacheable_attrs = uni.difference(reacheable_attrs)  # Atributos no alcanzables
    non_determinant_attrs = get_non_determinant(second_cover(fc), uni)  # No determinantes
    determinant_attrs = uni.difference(non_determinant_attrs)  # Determinantes
    keys = []  # Lista para almacenar las claves candidatas
    
    # Genera combinaciones de atributos determinantes
    for i in range(1, len(determinant_attrs) + 1):
        for comb in combinations(determinant_attrs, i):
            # Verifica si la combinación cumple con los requisitos de una clave
            if non_reacheable_attrs.issubset(comb) and not any(k.issubset(comb) for k in keys):
                closure_comb = closure_of_source_to_universe(comb, fc)  # Calcula el cierre
                if closure_comb == uni:  # Verifica si el cierre alcanza todo el universo
                    keys.append(set(comb))  # Agrega la clave candidata
    
    return keys  # Devuelve la lista de claves candidatas


# Genera un archivo de texto con la cobertura mínima en formato plano
def print_min_cover_3fn_to_txt(fc, sc, tc, tcgr, des, filename):
    """
    Genera un archivo de texto con la información de la cobertura mínima y la descomposición.

    Args:
        fc (list): Primer cubrimiento.
        sc (list): Segundo cubrimiento.
        tc (list): Tercer cubrimiento.
        tcgr (dict): Agrupación de dependencias.
        des (dict): Descomposición en 3FN.
        filename (str): Nombre del archivo de salida.
    """
    with open(filename, 'a') as file:
        file.write(f"First Cover: \n{cover_to_txt(fc)}\n\n")
        file.write(f"Second Cover: \n{cover_to_txt(sc)}\n\n")
        file.write(f"Third Cover: \n{cover_to_txt(tc)}\n\n")
        file.write(f"Group: \n{group_to_txt(tcgr)}\n\n")
        file.write(f"3FN: \n{decomposition_to_txt(des)}\n\n")


# Genera un archivo en formato LaTeX con la cobertura mínima
def print_min_cover_3fn_to_tex(fc, sc, tc, tcgr, des, filename):
    """
    Genera un archivo en formato LaTeX con la información de la cobertura mínima y la descomposición.

    Args:
        fc (list): Primer cubrimiento.
        sc (list): Segundo cubrimiento.
        tc (list): Tercer cubrimiento.
        tcgr (dict): Agrupación de dependencias.
        des (dict): Descomposición en 3FN.
        filename (str): Nombre del archivo de salida.
    """
    with open(filename, 'a') as file:
        file.write(f"First Cover: \n{cover_to_tex(fc)}\n\n")
        file.write(f"Second Cover: \n{cover_to_tex(sc)}\n\n")
        file.write(f"Third Cover: \n{cover_to_tex(tc)}\n\n")
        file.write(f"Group: \n{group_to_tex(tcgr)}\n\n")
        file.write(f"3FN: \n{decomposition_to_tex(des)}\n\n")


# Verifica la Propiedad de Join sin Pérdidas (PLJ) y la escribe en un archivo
def print_plj_to_txt(uni, fc, des, filename):
    """
    Verifica si una descomposición cumple con la PLJ y escribe el resultado en un archivo.

    Args:
        uni (list): Universo de atributos.
        fc (list): Lista de dependencias funcionales en la cobertura mínima.
        des (dict): Descomposición en 3FN.
        filename (str): Nombre del archivo de salida.
    """
    uni = list(uni)  # Convierte el universo a lista
    des = [des[i][0] for i in des]  # Extrae las relaciones de la descomposición
    fc = [[left, right[0]] for left, right in fc]  # Reorganiza las dependencias funcionales

    # Verifica y escribe el resultado de la PLJ
    print(f"\nPLJ: {plj.verificar_plj(uni, fc, des, filename)}")

# Convierte la lista de dependencias funcionales a texto plano (formato legible)
def cover_to_txt(df):
    """
    Genera una representación en texto plano de las dependencias funcionales.

    Args:
        df (list): Lista de dependencias funcionales (formato: [(X, A)]).

    Returns:
        str: Texto representando las dependencias funcionales en formato "X ⟶ A".
    """
    text = ""
    for l, r in df:
        # Combina los elementos del lado izquierdo (X) y derecho (A) con flechas
        text += ", ".join(l) + ' ⟶ ' + ", ".join(r) + '\n'
    return text  # Devuelve el texto generado


# Convierte la lista de dependencias funcionales a formato LaTeX
def cover_to_tex(df):
    """
    Genera una representación en LaTeX de las dependencias funcionales.

    Args:
        df (list): Lista de dependencias funcionales (formato: [(X, A)]).

    Returns:
        str: Representación de las dependencias funcionales en LaTeX.
    """
    text = ""
    for l, r in df:
        # Convierte la dependencia funcional a formato LaTeX usando flechas
        text += ", ".join(l) + ' $\\rightarrow$ ' + ", ".join(r) + ' \\newline \n'
    return text.replace('_', " ")  # Reemplaza guiones bajos para evitar conflictos en LaTeX


# Convierte las agrupaciones de dependencias funcionales a texto plano
def group_to_txt(df):
    """
    Genera una representación en texto plano de las agrupaciones de dependencias funcionales.

    Args:
        df (dict): Diccionario de dependencias agrupadas, donde las claves son conjuntos de determinantes.

    Returns:
        str: Texto representando las agrupaciones en formato "X ⟶ A".
    """
    text = ""
    for l in df:
        # Convierte las claves del diccionario y sus valores a texto plano
        text += ", ".join(l) + ' ⟶ ' + ", ".join(df[l]) + '\n'
    return text  # Devuelve el texto generado


# Convierte las agrupaciones de dependencias funcionales a formato LaTeX
def group_to_tex(df):
    """
    Genera una representación en LaTeX de las agrupaciones de dependencias funcionales.

    Args:
        df (dict): Diccionario de dependencias agrupadas, donde las claves son conjuntos de determinantes.

    Returns:
        str: Representación de las agrupaciones en LaTeX.
    """
    text = ""
    for l in df:
        # Convierte las claves y valores a formato LaTeX usando flechas
        text += ", ".join(l) + ' $\\rightarrow$ ' + ", ".join(df[l]) + ' \\newline \n'
    return text.replace('_', " ")  # Reemplaza guiones bajos para evitar conflictos en LaTeX


# Convierte la descomposición a texto plano
def decomposition_to_txt(df):
    """
    Genera una representación en texto plano de la descomposición.

    Args:
        df (dict): Diccionario de descomposiciones, donde cada entrada tiene el formato:
                   {i: (U, [(X, A)])}.

    Returns:
        str: Texto representando las descomposiciones.
    """
    text = ""
    for i in df:
        # Agrega el nombre de la relación, conjunto de atributos y dependencias funcionales
        text += f"R{i+1} ( U{i+1} , F{i+1} ) \n"
        text += f"U{i+1} = {df[i][0]}  \n".replace('\'', "")  # Elimina apóstrofos del conjunto de atributos
        text += f"F{i+1} = π_U{i+1} ≡ " + '{ '  # Representación en texto para dependencias
        for index, j in enumerate(df[i][1]):
            # Convierte cada dependencia funcional
            text += ' ⟶ '.join([', '.join(j[0]), ', '.join(j[1])])
            text += " ; " if index < len(df[i][1]) - 1 else " }"  # Agrega separadores
        text += "\n\n"
    return text  # Devuelve el texto generado


# Convierte la descomposición a formato LaTeX
def decomposition_to_tex(df):
    """
    Genera una representación en LaTeX de la descomposición.

    Args:
        df (dict): Diccionario de descomposiciones, donde cada entrada tiene el formato:
                   {i: (U, [(X, A)])}.

    Returns:
        str: Representación de las descomposiciones en LaTeX.
    """
    text = ""
    for i in df:
        under = "_" + "{" + f"{i+1}" + "}"  # Subíndices en LaTeX
        # Agrega la relación en LaTeX con sus atributos y dependencias
        text += f"$ R{under} ( U{under} , F{under} ) $ \\newline \n"
        text += f"$ U{under} = " + ("\{" + f"{df[i][0]}" + "\} $ \\newline \n") \
            .replace('\'', "") \
            .replace('_', " \\hspace{0.2cm} ") \
            .replace(',', ', \\hspace{0.2cm} ') \
            .replace('ñ', "\\tilde{n}")  # Representa caracteres especiales en LaTeX
        text += f"$ F{under} = \\sqcap_" + "{" + "U_" + "{" + f"{under}" + "}}(F) \\equiv " + '\{' 
        for index, j in enumerate(df[i][1]):
            # Representa las dependencias funcionales en LaTeX
            text += ' \\rightarrow '.join([', \\hspace{0.2cm} '.join(j[0]), ', \\hspace{0.2cm} '.join(j[1])]) \
                .replace('_', " \\hspace{0.2cm} ") \
                .replace('ñ', "\\tilde{n}")
            text += " \\hspace{0.2cm} ; \\hspace{0.2cm}" if index < len(df[i][1]) - 1 else " \\} $"
        text += "\\newline \n\n"
    return text  # Devuelve el texto generado

baseball = [
    "U-ID ⟶ Email, Password",
    "R-ID ⟶ TipoR",
    "U-ID ⟶ R-ID",
    "CI ⟶ NombreP, Edad, Apellidos",
    "BP-ID ⟶ CI",
    #"CI ⟶ BP-ID",
    "BP-ID ⟶ Promedio_de_bateo, Años_de_experiencia",
    "W-ID ⟶ CI",
    #"CI ⟶ W-ID",
    "W-ID ⟶ DT-ID",
    "DT-ID ⟶ W-ID",
    #"ED-ID ⟶ ED-ID",
    "E-ID ⟶ NombreE, Color, Entidad_representante, Iniciales",
    #"E-ID, AI-ID ⟶ E-ID, AI-ID",
    #"T-ID ⟶ T-ID",
    "T-ID, S-ID ⟶ NombreTS, TipoS, Fecha_de_inicio, Fecha_de_fin",
    "P-ID ⟶ NombrePos",
    "BP-ID ⟶ Lanzador",
    "Lanzador ⟶ BP-ID",
    "Lanzador ⟶ Mano_dominante, No_juegos_ganados, No_juegos_perdidos, Promedio_carreras",
    #"EC-ID ⟶ EC-ID",
    #"F-ID ⟶ F-ID",
    "W-ID ⟶ ED-ID", 
    "DT-ID ⟶ ED-ID",
    "ED-ID ⟶ DT-ID",
    "ED-ID ⟶ E-ID", 
    "E-ID ⟶ ED-ID",
    "T-ID, S-ID, E-ID ⟶ T-ID, S-ID, E-ID ",
    "T-ID, S-ID, BP-ID ⟶ E-ID",
    "T-ID, S-ID, P-ID ⟶ BP-ID",
    "BP-ID, P-ID ⟶ Efectividad",
    "E-ID, AI-ID, BP-ID, P-ID ⟶ E-ID, AI-ID, BP-ID, P-ID",
    "EC-ID ⟶ E-ID, AI-ID",
    "BP-ID, F-ID ⟶ BP-ID_2, P-ID",
    "BP-ID, F-ID ⟶ EC-ID",
    "EC-ID, F-ID ⟶ EC-ID_2, T-ID, S-ID",
    "EC-ID, F-ID ⟶ M-ID",
    "M-ID ⟶ E-ID, E-ID_2, Marcador_Ganador, Marcador_Perdedor",
]

#candidate_keys(baseball)

uni, fc, sc, tc, tcgr, des = min_cover_3fn(baseball)

print_min_cover_3fn_to_txt(fc, sc, tc, tcgr, des, 'Generate/baseball.txt')
print_min_cover_3fn_to_tex(fc, sc, tc, tcgr, des, 'Generate/baseball.tex')
print_plj_to_txt(uni, fc, des, 'Generate/baseball.txt')