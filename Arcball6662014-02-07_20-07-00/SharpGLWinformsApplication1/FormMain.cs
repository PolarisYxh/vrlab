using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using SharpGL.SceneGraph.Cameras;
using SharpGL.SceneGraph;
using SharpGL.SceneGraph.Lighting;
using SharpGL;
using SharpGL.SceneGraph.Core;
using SharpGL.SceneGraph.Assets;
using SharpGL.SceneGraph.Quadrics;
using SharpGL.SceneGraph.Effects;

namespace SharpGLWinformsApplication1
{
    public partial class FormMain : Form
    {
        private ArcBallEffect objectArcBallEffect;
        private ArcBallEffect axisArcBallEffect;

        public FormMain()
        {
            InitializeComponent();
            this.sceneControl1.MouseWheel += sceneControl1_MouseWheel;
        }

        void sceneControl1_MouseWheel(object sender, MouseEventArgs e)
        {
            objectArcBallEffect.ArcBall.Scale -= e.Delta * 0.001f;
        }
        const float near = 0.01f;
        const float far = 10000;
        private bool mouseDownFlag;

        private void InitElements(Scene scene)
        {
            var objectRoot = new SharpGL.SceneGraph.Primitives.Folder() { Name = "Root" };
            scene.SceneContainer.AddChild(objectRoot);
            // This implements free rotation(with translation and rotation).
            var camera = GetCamera();
            objectArcBallEffect = new ArcBallEffect(
                camera.Position.X, camera.Position.Y, camera.Position.Z,
                camera.Target.X, camera.Target.Y, camera.Target.Z,
                camera.UpVector.X, camera.UpVector.Y, camera.UpVector.Z);
            objectRoot.AddEffect(objectArcBallEffect);
            var axisRoot = new SharpGL.SceneGraph.Primitives.Folder() { Name = "axis root" };
            scene.SceneContainer.AddChild(axisRoot);
            axisArcBallEffect = new ArcBallEffect(camera.Position.X,
                camera.Position.Y, camera.Position.Z,
                camera.Target.X, camera.Target.Y, camera.Target.Z,
                camera.UpVector.X, camera.UpVector.Y, camera.UpVector.Z);
            axisRoot.AddEffect(axisArcBallEffect);

            InitLight(objectRoot);
            InitAxis(objectRoot);
            InitAxis(axisRoot);
            InitFrameElement(6, 24, 7, objectRoot);
            InitGridElement(1.5f, 3, 0, 3, 24, objectRoot);
        }

        private void InitGridElement(float x, float y, float z, float width, float length, SceneElement parent)
        {
            var folder = new SharpGL.SceneGraph.Primitives.Folder() { Name = "Grid" };
            parent.AddChild(folder);

            var grid = new GridElement(x, y, z, width, length);
            folder.AddChild(grid);
        }

        private void InitFrameElement(int width, int length, int height, SceneElement parent)
        {
            var folder = new SharpGL.SceneGraph.Primitives.Folder() { Name = "Frame" };
            parent.AddChild(folder);

            var frame = new FrameElement(width, length, height);
            folder.AddChild(frame);
        }

        private void InitAxis(SceneElement parent)
        {
            var folder = new SharpGL.SceneGraph.Primitives.Folder() { Name = "Axis" };
            parent.AddChild(folder);

            // X轴
            Material red = new Material();
            red.Emission = Color.Red;
            red.Diffuse = Color.Red;

            Cylinder x1 = new Cylinder() { Name = "X1" };
            x1.BaseRadius = 0.05;
            x1.TopRadius = 0.05;
            x1.Height = 1.5;
            x1.Transformation.RotateY = 90f;
            x1.Material = red;
            folder.AddChild(x1);

            Cylinder x2 = new Cylinder() { Name = "X2" };
            x2.BaseRadius = 0.1;
            x2.TopRadius = 0;
            x2.Height = 0.2;
            x2.Transformation.TranslateX = 1.5f;
            x2.Transformation.RotateY = 90f;
            x2.Material = red;
            folder.AddChild(x2);

            // Y轴
            Material green = new Material();
            green.Emission = Color.Green;
            green.Diffuse = Color.Green;

            Cylinder y1 = new Cylinder() { Name = "Y1" };
            y1.BaseRadius = 0.05;
            y1.TopRadius = 0.05;
            y1.Height = 1.5;
            y1.Transformation.RotateX = -90f;
            y1.Material = green;
            folder.AddChild(y1);

            Cylinder y2 = new Cylinder() { Name = "Y2" };
            y2.BaseRadius = 0.1;
            y2.TopRadius = 0;
            y2.Height = 0.2;
            y2.Transformation.TranslateY = 1.5f;
            y2.Transformation.RotateX = -90f;
            y2.Material = green;
            folder.AddChild(y2);

            // Z轴
            Material blue = new Material();
            blue.Emission = Color.Blue;
            blue.Diffuse = Color.Blue;

            Cylinder z1 = new Cylinder() { Name = "Z1" };
            z1.BaseRadius = 0.05;
            z1.TopRadius = 0.05;
            z1.Height = 1.5;
            z1.Material = blue;
            folder.AddChild(z1);

            Cylinder z2 = new Cylinder() { Name = "Z2" };
            z2.BaseRadius = 0.1;
            z2.TopRadius = 0;
            z2.Height = 0.2;
            z2.Transformation.TranslateZ = 1.5f;
            z2.Material = blue;
            folder.AddChild(z2);
        }

        private void InitLight(SceneElement parent)
        {
            Light light1 = new Light()
            {
                Name = "Light 1",
                On = true,
                Position = new Vertex(-9, -9, 11),
                GLCode = OpenGL.GL_LIGHT0
            };
            Light light2 = new Light()
            {
                Name = "Light 2",
                On = true,
                Position = new Vertex(9, -9, 11),
                GLCode = OpenGL.GL_LIGHT1
            };
            Light light3 = new Light()
            {
                Name = "Light 3",
                On = true,
                Position = new Vertex(0, 15, 15),
                GLCode = OpenGL.GL_LIGHT2
            };
            var folder = new SharpGL.SceneGraph.Primitives.Folder() { Name = "Lights" };

            parent.AddChild(folder);
            folder.AddChild(light1);
            folder.AddChild(light2);
            folder.AddChild(light3);
        }

        private void sceneControl1_MouseDown(object sender, MouseEventArgs e)
        {
            this.mouseDownFlag = true;
            objectArcBallEffect.ArcBall.SetBounds(this.sceneControl1.Width, this.sceneControl1.Height);
            objectArcBallEffect.ArcBall.MouseDown(e.X, e.Y);
            axisArcBallEffect.ArcBall.SetBounds(this.sceneControl1.Width, this.sceneControl1.Height);
            axisArcBallEffect.ArcBall.MouseDown(e.X, e.Y);
        }

        private void sceneControl1_MouseMove(object sender, MouseEventArgs e)
        {
            objectArcBallEffect.ArcBall.MouseMove(e.X, e.Y);
            axisArcBallEffect.ArcBall.MouseMove(e.X, e.Y);
        }

        private void sceneControl1_MouseUp(object sender, MouseEventArgs e)
        {
            this.mouseDownFlag = false;
            objectArcBallEffect.ArcBall.MouseUp(e.X, e.Y);
            axisArcBallEffect.ArcBall.MouseUp(e.X, e.Y);
        }

        private void sceneControl1_KeyDown(object sender, KeyEventArgs e)
        {
            const float interval = 1;
            if (e.KeyCode == Keys.W || e.KeyCode == Keys.Up)
            {
                this.objectArcBallEffect.ArcBall.GoUp(interval);
            }
            else if (e.KeyCode == Keys.S || e.KeyCode == Keys.Down)
            {
                this.objectArcBallEffect.ArcBall.GoDown(interval);
            }
            else if (e.KeyCode == Keys.A || e.KeyCode == Keys.Left)
            {
                this.objectArcBallEffect.ArcBall.GoLeft(interval);
            }
            else if (e.KeyCode == Keys.D || e.KeyCode == Keys.Right)
            {
                this.objectArcBallEffect.ArcBall.GoRight(interval);
            }
        }

        private LookAtCamera GetCamera()
        {
            return this.sceneControl1.Scene.CurrentCamera as LookAtCamera;
        }

        private void FormMain_Resize(object sender, EventArgs e)
        {
            this.objectArcBallEffect.ArcBall.SetBounds(this.sceneControl1.Width, this.sceneControl1.Height);
            var gl = this.sceneControl1.OpenGL;
            var axis = gl.UnProject(50, 50, 0.1);
            axisArcBallEffect.ArcBall.SetTranslate(axis[0], axis[1], axis[2]);
            axisArcBallEffect.ArcBall.Scale = 0.001f;
        }

        private void sceneControl1_OpenGLInitialized(object sender, EventArgs e)
        {
            var scene = this.sceneControl1.Scene;
            scene.SceneContainer.Children.Clear();
            scene.RenderBoundingVolumes = false;
            // 设置视角
            var camera = GetCamera();
            camera.Near = near;
            camera.Far = far;
            camera.Position = new Vertex(12.5f, -1.5f, 11.5f);
            camera.Target = new Vertex(4.5f, 7, 2.5f);
            camera.UpVector = new Vertex(0.000f, 0.000f, 1.000f);

            InitElements(scene);
            axisArcBallEffect.ArcBall.SetTranslate(
                12.490292456095853f, -1.5011389593859834f, 11.489356270615454f);
            axisArcBallEffect.ArcBall.Scale = 0.001f;
        }

    }
}
