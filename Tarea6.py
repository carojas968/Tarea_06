#Tarea No. 6 CRISTHIAN ALEJANDRO ROJAS MARTINEZ
import cv2
from enum import Enum
import numpy as np
import sys
import os

#Extraer datos de una ruta
from pathlib import Path
def ls3(path):
    return [obj.name for obj in Path(path).iterdir() if obj.is_file()]



class Methods(Enum):
    SIFT = 1
    ORB = 2


if __name__ == '__main__':

    path = sys.argv[1]




    # Contar el numero de fotos en la ruta 'path'
    a = ls3(path)
    image = []
    total = 0
    for image_name in a:
        path_file = os.path.join(path, image_name)
        imagen = cv2.imread(path_file)
        image.append(imagen)
        total += 1
    # Imprime el total de imagenes a trabajar
    print("El total de imagenes para el ejercicio es de", total)
    # Solicita al usuario el numero de imagen de referencia
    ref = int(input("Ingrese el numero de imagen de referencia: "))
    # Solicita al usuario el tipo de metodo para deteccion de puntos
    TipoMetodo = int(input("Digite el metodo de deteccion de púntos: 1) SIFT  2) ORB "))

    # sift/orb interest points and descriptors
    if TipoMetodo == 1:
        method = Methods.SIFT
    else:
        method = Methods.ORB





# Metodo que permite generar la homografia apartir de un par de imagenes concatenadas; tiene dos parametros 1. la imagen concatenada 2. un arreglo de puntos inicializados en vacio retorna la matriz de homografia
def generarHomografia( image_1, image_2, method):
    image_gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    if method == Methods.SIFT:
        sift = cv2.SIFT_create(nfeatures=100)  # shift invariant feature transform
        keypoints_1, descriptors_1 = sift.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_2, None)
    else:
        orb = cv2.ORB_create(nfeatures=100)  # oriented FAST and Rotated BRIEF
        keypoints_1, descriptors_1 = orb.detectAndCompute(image_gray_1, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_2, None)

    # image_draw_1 = cv2.drawKeypoints(image_gray_1, keypoints_1, None)
    # image_draw_2 = cv2.drawKeypoints(image_gray_2, keypoints_2, None)

        # image_draw = cv2.hconcat(np.copy(image[0]),np.copy(image[1]))
    # Interest points matching
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
    image_matching = cv2.drawMatchesKnn(image_1, keypoints_1, image_2, keypoints_2, matches, None)



    # Retrieve matched points
    points_1 = []
    points_2 = []
    for idx, match in enumerate(matches):
        idx2 = match[0].trainIdx
        points_1.append(keypoints_1[idx].pt)
        points_2.append(keypoints_2[idx2].pt)

    # Compute homography and warp image_1
    H, _ = cv2.findHomography(np.array(points_1), np.array(points_2), method=cv2.RANSAC)

    return H
# ---------- GENERACION DE TODOS LOS PARES DE FOTOGRAFIAS Y PROCESAMIENTO PARA OBTENER LAS HOMGRAFIAS POR PARES -----------------------------------------------------

# para cada imagen desde la 0  generar un par de imagen y obtener la homografia
Homografias = [] # En este arreglo se almacenan las homografias resultado de la concatenacion
for i in range(1,total-1):

    a = generarHomografia(image[i-1], image[i], method)
    Homografias.append(a)


print(len(Homografias))
# Encontrar las homografias basados en la imagen referencia indicada por el usuario  ______OK
ref = ref - 1
resultados = []  # Alamcena las proyecciones de las homgrafias en relacion a la imagen de referencia
for i in range(0, len(Homografias)):
    # genera las tranformaciones de la primera mitad de utilizando la imagen de referencia como la mitad
    if (ref) > i:
        # Si las homografias estan a un paso de la imagen de refenrencia
        if i - (ref - 1) == 0:
            resultados.append(Homografias[i]) # en reultado se debe almacenar el resultado de la homografia
        else: # si la homografia esta a mas de un paso de la imagen de refenrencia
            contador = i
            mult = Homografias[contador]
            while contador != ref -1:
                mult = mult * Homografias[contador + 1]
                contador += 1
            resultados.append(mult)
    # genera las tranformaciones de la segunda mitad de utilizando la imagen de referencia como la mitad
    else:
        if i - (ref) == 0: # si la homografia esta a un paso de la imagen de referencia
            resultados.append(np.linalg.inv(Homografias[i]))
        else: # si la homografia esta a más de un paso de la imgen de referencia
            contador = i
            mult = Homografias[contador]
            while contador != ref:
                mult = mult * Homografias[contador - 1]
                contador -= 1
            a = np.linalg.inv(mult)
            resultados.append(a)
print(len(resultados))

# Seleccion de imagenes a aproyectara
imagenesP = []
for i in range(0,len(image)):
    if i != ref:
        imagenesP.append(image[i])
wrappedimages = []
for k in range(0, len(imagenesP)):
    a = cv2.warpPerspective(imagenesP[k], resultados[k], (imagenesP[k].shape[1], imagenesP[k].shape[0]))
    wrappedimages.append(a)
    cv2.imshow("Image warped", a)
cv2.waitKey(0)
#generar Promedio
f = []
for t in wrappedimages:
    f.append(np.where( t == 0, np.nan, t ))
    #t[t == 0.0] = np.nan

x = image[0]

cv2.imshow("Image warped", x)
cv2.waitKey(0)