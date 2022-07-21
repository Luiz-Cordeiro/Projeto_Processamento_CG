## Importação das bibliotecas

import cv2   
import numpy as np

## Carregamento da imagem
def loadImage(image: str):
    image = cv2.imread(image, cv2.IMREAD_COLOR) ## CHAVE ALTO CONSTRASTE 
    return image

## Conversão para escala de cinza
def toGrayScale(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayScale

def gaussFilter(image):
    gaussFilteredImage = cv2.medianBlur(image, 1)
    return gaussFilteredImage

## Controle de brilho e contraste
## Expressão obtida pelo trabalho disponível em: https://codestack.club/questions/1935302/brilho-e-contraste-do-opencv-como-no-gimp
def gimpBrightContrastControl(image, brightness: float, contrast: float):

    brightness = brightness 
    contrast = contrast
    shadow = 0
    highlight = 255 + brightness
    alpha_b = (highlight - shadow)/255 
    gamma_b = shadow
    ## Sobreposição de duas imagens iguais (com arrays de pesos diferentes) para alterar o brilho
    brightContrastProcImage = cv2.addWeighted(image, alpha_b, image, 0, gamma_b) 
    ## Conversão para escala [-127,127] do constraste
    f = 131*(contrast + 127)/(127*(131-contrast)) 
    alpha_c = f
    gamma_c = 127*(1-f)
    ## Sobreposição de duas imagens iguais (com arrays de pesos diferentes) para alterar o constraste
    brightContrastProcImage = cv2.addWeighted(brightContrastProcImage, alpha_c, brightContrastProcImage, 0, gamma_c)
    return brightContrastProcImage

## Aplicação do algoritmo Sobel para detectar bordas no sentido horizontal
def applySobelGradientX(image, scale, delta, ddepth):   
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    return grad_x

## Aplicação do algoritmo Sobel para detectar bordas no sentido vertical
def applySobelGradientY(image, scale, delta, ddepth):   
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    return grad_y

## Justaposição as imagens (kernel Sobel vertical e horizontal) para conjugar o resultado da detecção de bordas
def mergeGradients(grad_x, grad_y):
    abs_grad_x = cv2.convertScaleAbs(grad_x) # Converte a imagem para 8 bits
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    imgSobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) # Pondera os pesos das matrizes igualmente
    return imgSobel

## Aplicação do Threshold
def applyThreshold(image, minThres: float, maxThres: float):
    ret, imgThreshold = cv2.threshold(image, minThres, maxThres, cv2.THRESH_BINARY)
    return imgThreshold

## Aplicação da transformada de Hough probabilística (para detectar linhas)
def applyHoughTransformLines(image, threshold, minLineLength, maxLineGap):
    lines = cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=minLineLength,maxLineGap = maxLineGap) ## COM THRESHOLD
    return lines

# Desenho das linhas detectadas sobre a imagem original
def drawLines(image, lines):
    ## Bloco try-except para tratar possível retorno nulo da detecção de linhas
    try:
        if(lines.any != None):

            for line in lines:
                x1, y1, x2, y2 = line[0]
                ## Usa dois pontos para criar a reta, a cor (padrão BGR) e a espessura da linha desenhada
                cv2.line(image, (x1, y1), (x2, y2), (0,0,255),4)
            return len(lines)
    except:
        #print("Nenhuma linha foi identificada nesta imagem")
        return 0

## Aplicação da transformada de Hough (para detectar círculos)
def applyHoughTransformCircles(image, minRadius, maxRadius):
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                param1=200, param2=34,
                                minRadius=minRadius, maxRadius=maxRadius)
    return circles

## Desenho dos círculos detectados sobre a imagem original
def drawCircles(image, circles):

    try:
        if (circles.any != None):
            circlesDetected = np.uint16(np.around(circles))
            for i in circlesDetected[0, :]:
                center = (i[0], i[1])
                cv2.circle(image, center, 1, (0, 100, 100), 3) ## Desenho do centro do círculo
                # circle outline
                radius = i[2]
                cv2.circle(image, center, radius, (0, 0, 255), 3) ## Desenho da circunferência
            return len(circles)
    except:
        return 0

## Salvamento da imagem final
def saveDetectedForeignBodies(imageName:str, image, nroCircles, nroLines):
    if(nroCircles !=0 or nroLines != 0):
        return cv2.imwrite('./Detected/DETECTED_{}'.format(imageName),image) ## Salva na pasta de deteções
    else:
        return cv2.imwrite('./NotDetected/NOT_DETECTED_{}'.format(imageName),image) ## Salva na pasta onde não há detecções



def main():
    ## Vetor com os nomes dos arquivos de teste
    imagens = ["rx_1.png", "rx_2.png", "rx_3.png", "rx_4.png", "rx_5.png",
           "rx_6.png", "rx_7.png", "rx_8.png", "rx_9.png", "rx_10.png", 
           "rx_11.png", "rx_12.png", "rx_13.png", "rx_14.png", "rx_15.png", 
           "rx_16.png", "rx_17.png", "rx_18.png"]
    
    for i in range (0, len(imagens)):
        loadedImage = loadImage(imagens[i])
        grayImage = toGrayScale(loadedImage)
        imageGauss = gaussFilter(grayImage)
        adjustedContrastBrightImg = gimpBrightContrastControl(imageGauss, -100, 40)
        xSobelGradient = applySobelGradientX(adjustedContrastBrightImg, 1, 0, cv2.CV_16UC1)
        ySobelGradient = applySobelGradientY(adjustedContrastBrightImg, 1, 0, cv2.CV_16UC1)
        imgSobel = mergeGradients(xSobelGradient, ySobelGradient)
        imageThreshold = applyThreshold(imgSobel, 121, 255)
        detectedLines = applyHoughTransformLines(imageThreshold, 30, 3, 1)
        nroDetectedLines = drawLines(loadedImage, detectedLines)
        detectedCircles = applyHoughTransformCircles(adjustedContrastBrightImg, 0, 70)
        nroDetectedCircles = drawCircles(loadedImage, detectedCircles)
        print("{} -> Círculos: {} | Linhas: {}".format(imagens[i], nroDetectedCircles, nroDetectedLines))
        saveDetectedForeignBodies(imagens[i], loadedImage, nroDetectedCircles, nroDetectedLines)

    print("Detecção finalizada.")
    input("PRESSIONE ENTER PARA SAIR")

## Módulo para execução da função principal
if __name__ == "__main__":
    main()