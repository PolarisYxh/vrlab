#ifndef VBOTORUS_H
#define VBOTORUS_H


class VBOTorus 
{
private:
    unsigned int vaoHandle;
    int faces, rings, sides;

    void generateVerts(float * , float * ,float *, unsigned int *,
                       float , float);

public:
    VBOTorus(float, float, int, int);

    void render() const;

	int getVertexArrayHandle();
};

#endif // VBOTORUS_H
