using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpGL.SceneGraph.Core;
using SharpGL;

namespace SharpGLWinformsApplication1
{
    class GridElement:SceneElement
        ,IRenderable
    {
        private float x;
        private float y;
        private float z;
        private float width;
        private float length;
        public GridElement(float x, float y, float z, float width, float length)
        {
            this.x = x; this.y = y; this.z = z;
            this.width = width; this.length = length;
        }
        public void Render(SharpGL.OpenGL gl, RenderMode renderMode)
        {
            gl.PushAttrib(OpenGL.GL_CURRENT_BIT | OpenGL.GL_ENABLE_BIT | OpenGL.GL_LINE_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.Disable(OpenGL.GL_LIGHTING);

            gl.LineWidth(1.0f);

            gl.Begin(OpenGL.GL_LINES);

            // 边框
            gl.Color(0.6f, 0.6f, 0.6f);

            gl.Vertex(x, y, z);
            gl.Vertex(x + width, y, z);

            gl.Vertex(x + width, y, z);
            gl.Vertex(x + width, y + length, z);

            gl.Vertex(x + width, y + length, z);
            gl.Vertex(x, y + length, z);

            gl.Vertex(x, y + length, z);
            gl.Vertex(x, y, z);

            // 横向
            for (var i = 1; i < width+ 1; i++)
            {
                if (i==width) break;

                gl.Color(0.6f, 0.6f, 0.6f);

                gl.Vertex(x + i, y, z);
                gl.Vertex(x + i, y + length, z);
            }

            for (int i = 1; i < width * 10 + 1; i++)
            {
                if (i==width * 10) break;

                if (i % 10 != 0)
                {
                    gl.Color(0.3f, 0.3f, 0.3f);

                    gl.Vertex(x + i / 10f, y, z);
                    gl.Vertex(x + i / 10f, y + length, z);
                }
            }

            // 竖向
            for (int i = 1; i < length + 1; i++)
            {
                if (i==length) break;

                gl.Color(0.6f, 0.6f, 0.6f);

                gl.Vertex(x, y + i, z);
                gl.Vertex(x + width, y + i, z);
            }

            for (int i = 1; i < length * 10 + 1; i++)
            {
                if (i==length * 10) break;

                if (i % 10 != 0)
                {
                    gl.Color(0.3f, 0.3f, 0.3f);

                    gl.Vertex(x, y + i / 10f, z);
                    gl.Vertex(x + width, y + i / 10f, z);
                }
            }

            gl.End();

            gl.PopAttrib();
        }
    }
}
