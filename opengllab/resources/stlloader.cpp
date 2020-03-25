#include <fstream>
#include <string>
#include <vector>
bool ReadSTLFile(const char* cfilename)
{
	if (cfilename == NULL)
		return false;

	std::ifstream in(cfilename, std::ifstream::in);

	if (!in)
		return false;

	std::string headStr;
	getline(in, headStr, ' ');
	in.close();

	if (headStr.empty())
		return false;

	if (headStr[0] == 's')
	{
		ReadASCII(cfilename);
	}
	else
	{
		ReadBinary(cfilename);
	}
	return true;
}

bool ReadASCII(const char* cfilename)
{
	std::vector<float> coorX;
	std::vector<float> coorY;
	std::vector<float> coorZ;

	int i = 0, j = 0, cnt = 0, pCnt = 4;
	char a[100];
	char str[100];
	double x = 0, y = 0, z = 0;

	std::ifstream in(cfilename, std::ifstream::in);

	if (!in)
		return false;
	do
	{
		i = 0;
		cnt = 0;
		in.getline(a, 100, '\n');
		while (a[i] != '\0')
		{
			if (!islower((int)a[i]) && !isupper((int)a[i]) && a[i] != ' ')
				break;
			cnt++;
			i++;
		}

		while (a[cnt] != '\0')
		{
			str[j] = a[cnt];
			cnt++;
			j++;
		}
		str[j] = '\0';
		j = 0;

		if (sscanf(str, "%lf%lf%lf", &x, &y, &z) == 3)
		{
			coorX.push_back(x);
			coorY.push_back(y);
			coorZ.push_back(z);
		}
		pCnt++;
	} while (!in.eof());

	return true;
}

bool ReadBinary(const char* cfilename)
{
	std::vector<float> coorX;
	std::vector<float> coorY;
	std::vector<float> coorZ;

	char str[80];
	std::ifstream in(cfilename, std::ifstream::in | std::ifstream::binary);

	if (!in)
		return false;

	in.read(str, 80);

	//number of triangles  
	int triangles;
	in.read((char*)&triangles, sizeof(int));

	if (triangles == 0)
		return false;

	for (int i = 0; i < triangles; i++)
	{
		float coorXYZ[12];
		in.read((char*)coorXYZ, 12 * sizeof(float));

		for (int j = 1; j < 4; j++)
		{
			coorX.push_back(coorXYZ[j * 3]);
			coorY.push_back(coorXYZ[j * 3 + 1]);
			coorZ.push_back(coorXYZ[j * 3 + 2]);
		}

		in.read((char*)coorXYZ, 2);
	}

	in.close();
	return true;
}