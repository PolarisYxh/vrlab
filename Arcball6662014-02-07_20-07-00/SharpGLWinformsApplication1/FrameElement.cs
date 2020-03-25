using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpGL.SceneGraph.Core;
using SharpGL;

namespace SharpGLWinformsApplication1
{
    class FrameElement:SceneElement,
        IRenderable
    {
        private float width;
        private float length;
        private float height;
        public FrameElement(float width, float length, float height)
        {
            this.width = width; this.length = length; this.height = height;
        }
        public void Render(SharpGL.OpenGL gl, RenderMode renderMode)
        {
            gl.PushAttrib(OpenGL.GL_CURRENT_BIT | OpenGL.GL_ENABLE_BIT | OpenGL.GL_LINE_BIT | OpenGL.GL_DEPTH_BUFFER_BIT);
            gl.Disable(OpenGL.GL_LIGHTING);

            gl.LineWidth(1.0f);

            gl.Begin(OpenGL.GL_LINES);

            // 下底面
            gl.Color(0.3f, 0.3f, 0.3f);

            gl.Vertex(0, 0, 0);
            gl.Vertex(width, 0, 0);

            gl.Vertex(width, 0, 0);
            gl.Vertex(width, length, 0);

            gl.Vertex(width, length, 0);
            gl.Vertex(0, length, 0);

            gl.Vertex(0, length, 0);
            gl.Vertex(0, 0, 0);

            // 上顶面
            gl.Color(0.3f, 0.3f, 0.3f);

            gl.Vertex(0, 0, height);
            gl.Vertex(width, 0, height);

            gl.Vertex(width, 0, height);
            gl.Vertex(width, length, height);

            gl.Vertex(width, length, height);
            gl.Vertex(0, length, height);

            gl.Vertex(0, length, height);
            gl.Vertex(0, 0, height);

            // 右表面
            gl.Color(0.3f, 0.3f, 0.3f);

            gl.Vertex(0, 0, 0);
            gl.Vertex(0, 0, height);

            gl.Vertex(width, 0, 0);
            gl.Vertex(width, 0, height);

            // 左表面
            gl.Color(0.3f, 0.3f, 0.3f);

            gl.Vertex(0, length, 0);
            gl.Vertex(0, length, height);

            gl.Vertex(width, length, 0);
            gl.Vertex(width, length, height);

            gl.End();

            gl.PopAttrib();
        }
    }
}
