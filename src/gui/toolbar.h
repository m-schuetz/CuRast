

// see https://github.com/ocornut/imgui/issues/2648
void CuRast::makeToolbar(){

	auto editor = CuRast::instance;
	auto& scene = editor->scene;
	auto drawlist = ImGui::GetForegroundDrawList();

	ImVec2 toolbar_start = ImVec2(0, 19);

	ImGui::SetNextWindowPos(toolbar_start);
	ImVec2 requested_size = ImVec2(VKRenderer::width, 0.0f);
	ImGui::SetNextWindowSize(requested_size);

	struct Section{
		float x_start;
		float x_end;
		float y_start;
		float y_end;
		string label;
		ImVec2 CurrLineSize;
		ImVec2 PrevLineSize;
	};

	vector<Section> sections;

	auto startSection = [&](string label){
		// ImGui::SameLine();

		ImGuiWindow* window = ImGui::GetCurrentWindow();

		Section section;
		section.x_start = ImGui::GetCursorPosX();
		section.y_start = ImGui::GetCursorPosY();
		section.CurrLineSize = window->DC.CurrLineSize;
		section.PrevLineSize = window->DC.PrevLineSize;
		section.label = label;

		sections.push_back(section);

		ImGui::BeginGroup();
	};

	auto endSection = [&](){
		ImGui::EndGroup();

		ImGui::SameLine();
		float x = ImGui::GetCursorPosX();
		ImU32 color = IM_COL32(255, 255, 255, 75);
		drawlist->AddLine({x - 4.0f, 51.0f - 32.0f}, {x - 4.0f, 120.0f - 32.0f}, color, 1.0f);

		Section& section = sections[sections.size() - 1];
		section.x_end = x;

		ImGuiWindow* window = ImGui::GetCurrentWindow();
		window->DC.CurrLineSize = section.CurrLineSize;
		window->DC.PrevLineSize = section.PrevLineSize;
	};

	auto startHighlightButtonIf = [&](bool condition){
		ImGuiStyle* style = &ImGui::GetStyle();
		ImVec4* colors = style->Colors;
		ImVec4 color = colors[ImGuiCol_Button];

		if(condition){
			color = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
		}
		ImGui::PushStyleColor(ImGuiCol_Button, color);
	};

	auto endHighlightButtonIf = [&](){
		ImGui::PopStyleColor(1);
	};
	
	uint32_t flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
	ImGui::Begin("Toolbar", nullptr, flags);

	
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.0f, 0.0f, 0.0f, 0.0f});
	{
		ImVec2 buttonSize = ImVec2(32.0f, 32.0f);
		float symbolSize = 32.0f;
		ImVec4 bg_col = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		ImVec4 tint_col = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
		float iconSize = 16.0f;

		if (ImGui::BeginTable("ToolbarTable", 3, ImGuiTableFlags_SizingFixedFit)){

			{ // WIDGETS

				ImGui::TableNextRow();
				ImGui::TableSetColumnIndex(0);
				// ImGui::AlignTextToFramePadding();
				startSection("Widgets");

				// ImGui::SameLine();
				startHighlightButtonIf(CuRastSettings::showKernelInfos);
				if(ImGui::Button("Kernels")){
					CuRastSettings::showKernelInfos = !CuRastSettings::showKernelInfos;
				}
				endHighlightButtonIf();

				ImGui::SameLine();
				startHighlightButtonIf(CuRastSettings::showMemoryInfos);
				if(ImGui::Button("Memory")){
					CuRastSettings::showMemoryInfos = !CuRastSettings::showMemoryInfos;
				}
				endHighlightButtonIf();

				// ImGui::SameLine();
				startHighlightButtonIf(CuRastSettings::showTimingInfos);
				if(ImGui::Button("Timings")){
					CuRastSettings::showTimingInfos = !CuRastSettings::showTimingInfos;
				}
				endHighlightButtonIf();

				ImGui::SameLine();
				startHighlightButtonIf(CuRastSettings::showStats);
				if(ImGui::Button("Stats")){
					CuRastSettings::showStats = !CuRastSettings::showStats;
				}
				endHighlightButtonIf();

				startHighlightButtonIf(CuRastSettings::showOverlay);
				if(ImGui::Button("Overlay")){
					CuRastSettings::showOverlay = !CuRastSettings::showOverlay;
				}
				endHighlightButtonIf();

				ImGui::SameLine();
				startHighlightButtonIf(CuRastSettings::showBenchmarking);
				if(ImGui::Button("Benchmarking")){
					CuRastSettings::showBenchmarking = !CuRastSettings::showBenchmarking;
				}
				endHighlightButtonIf();

				endSection();
			}
			

			{ // DEV
				ImGui::TableSetColumnIndex(1);

				startSection("Dev");

				ImGui::Checkbox("bounding boxes", &CuRastSettings::showBoundingBoxes);
				ImGui::SameLine();
				ImGui::Checkbox("frustum culling", &CuRastSettings::enableFrustumCulling);
				ImGui::SameLine();
				ImGui::Checkbox("freeze frustum", &CuRastSettings::freezeFrustum);

				// ImGui::SameLine();
				ImGui::SetNextItemWidth(200.0f);
				ImGui::SliderFloat("threshold", &CuRastSettings::threshold, 0.0f, 1.0f);
				
				ImGui::SameLine();
				ImGui::Checkbox("Disable Instancing", &CuRastSettings::disableInstancing);

				ImGui::Checkbox("Enable Picking", &CuRastSettings::enableObjectPicking);
				
				// ImGui::SameLine();
				// string strMeasure;
				// if(CuRastSettings::measurementCountdown >= 0){
				// 	strMeasure = format("Measure 60 frames ({:2})", CuRastSettings::measurementCountdown);
				// }else{
				// 	strMeasure = "Measure 60 frames";
				// }
				// if(ImGui::Button(strMeasure.c_str())){
				// 	CuRastSettings::measurementCountdown = 60;
				// }


				endSection();

			}

			{ // MISC
				ImGui::TableSetColumnIndex(2);

				startSection("Misc");

				if(ImGui::Button("Copy Camera")){

					auto pos = Runtime::controls->getPosition();
					auto target = Runtime::controls->target;

					stringstream ss;
					ss << format("// position: {}, {}, {} \n", pos.x, pos.y, pos.z);
					ss << format("Runtime::controls->yaw    = {:.3f};\n", Runtime::controls->yaw);
					ss << format("Runtime::controls->pitch  = {:.3f};\n", Runtime::controls->pitch);
					ss << format("Runtime::controls->radius = {:.3f};\n", Runtime::controls->radius);
					ss << format("Runtime::controls->target = {{ {:.3f}, {:.3f}, {:.3f}}};\n", target.x, target.y, target.z);

					string str = ss.str();

					glfwSetClipboardString(nullptr, str.c_str());
				}

				ImGui::SameLine();
				if(ImGui::Button("Screenshot")){
					CuRastSettings::requestScreenshot = make_shared<string>("");
				}
				
				ImGui::Text("Rasterizer CUDA: ");
				static int mode = CuRastSettings::rasterizer;

				ImGui::SameLine();
				ImGui::RadioButton("Visbuffer[indexed]##rasterizer", &mode, RASTERIZER_VISBUFFER_INDEXED); 
				ImGui::SameLine();
				ImGui::RadioButton("Visbuffer[instanced]##rasterizer", &mode, RASTERIZER_VISBUFFER_INSTANCED); 

				#ifdef USE_VULKAN_SHARED_MEMORY
					ImGui::Text("Rasterizer Vulkan: ");

					ImGui::SameLine();
					ImGui::RadioButton("Indexed Draw##rasterizer", &mode, RASTERIZER_VULKAN_INDEXED_DRAW); 
					// if(ImGui::IsItemHovered())ImGui::SetTooltip("Vulkan Forward Indexed Draw");

					ImGui::SameLine();
					ImGui::RadioButton("Index-Pulling##rasterizer", &mode, RASTERIZER_VULKAN_INDEXPULLING_INSTANCED); 
					// if(ImGui::IsItemHovered())ImGui::SetTooltip("Vulkan VisibilityBuffer IndexPulling");

					ImGui::SameLine();
					ImGui::RadioButton("Visbuffer##rasterizer", &mode, RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER); 
					// if(ImGui::IsItemHovered())ImGui::SetTooltip("Vulkan VisibilityBuffer IndexPulling");
				#else 
					ImGui::BeginDisabled(true);
					ImGui::Text("Rasterizer Vulkan: Disabled. Compile with USE_VULKAN_SHARED_MEMORY");
					ImGui::EndDisabled();
					// if(ImGui::IsItemHovered())ImGui::SetTooltip("Requires enabling USE_VULKAN_SHARED_MEMORY in CuRastSettings.h");

					// ImGui::BeginDisabled(true);

					// ImGui::SameLine();
					// ImGui::RadioButton("FORWARD##rasterizer", &mode, RASTERIZER_VULKAN_INDEXED_DRAW);

					// ImGui::SameLine();
					// ImGui::RadioButton("VISBUFFER##rasterizer", &mode, RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER); 

					// ImGui::EndDisabled();
				#endif


				CuRastSettings::rasterizer = mode;

				// ImGui::SameLine();
				endSection();
			}

			{ // Appearance
				ImGui::TableSetColumnIndex(2);

				startSection("Appearance");

				ImGui::Text("Attribute: ");
				ImGui::SameLine();
				ImGui::RadioButton("None##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::NONE); 
				ImGui::SameLine();
				ImGui::RadioButton("Texture##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::TEXTURE); 
				ImGui::SameLine();
				ImGui::RadioButton("VertexColor##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::VERTEX_COLORS); 
				ImGui::SameLine();
				ImGui::RadioButton("UV##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::UV); 
				ImGui::SameLine();
				ImGui::RadioButton("Normal##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::NORMAL); 
				ImGui::SameLine();
				ImGui::RadioButton("triangleID##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::TRIANGLE_ID); 
				ImGui::SameLine();
				ImGui::RadioButton("meshID##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::MESH_ID); 
				ImGui::SameLine();
				ImGui::RadioButton("Stage##displayAttribute", (int*)&CuRastSettings::displayAttribute, (int)DisplayAttribute::STAGE); 

				ImGui::Checkbox("EDL", &CuRastSettings::enableEDL);
				ImGui::SameLine();
				ImGui::Checkbox("SSAO", &CuRastSettings::enableSSAO);
				ImGui::SameLine();
				ImGui::Checkbox("Diffuse", &CuRastSettings::enableDiffuseLighting);
				ImGui::SameLine();
				ImGui::Checkbox("Wireframe", &CuRastSettings::showWireframe);
				
				ImGui::SameLine();
				ImGui::Text("Background:");
				ImGui::SameLine();
				static int bg = 1;
				ImGui::RadioButton("blueish##background", &bg, 1);
				ImGui::SameLine();
				ImGui::RadioButton("white##background", &bg, 2);
				ImGui::SameLine();
				ImGui::RadioButton("black##background", &bg, 3);
				if(bg == 1) CuRastSettings::background = {0.3f, 0.4f, 0.5f, 1.0f};
				if(bg == 2) CuRastSettings::background = {1.0f, 1.0f, 1.0f, 1.0f};
				if(bg == 3) CuRastSettings::background = {0.0f, 0.0f, 0.0f, 1.0f};
				
				
				ImGui::Text("Supersampling: ");
				if(ImGui::IsItemHovered()) ImGui::SetTooltip("Number of samples per pixel. Increases framebuffer sizes accordingly.");
				ImGui::SameLine();
				ImGui::RadioButton("No##supersampling", (int*)&CuRastSettings::supersamplingFactor, 1);
				ImGui::SameLine();
				ImGui::RadioButton("2x2##supersampling", (int*)&CuRastSettings::supersamplingFactor, 2);
				ImGui::SameLine();
				ImGui::RadioButton("4x4##supersampling", (int*)&CuRastSettings::supersamplingFactor, 4);
				
				endSection();
			}

			{ // Actions
				ImGui::TableSetColumnIndex(2);

				startSection("Actions");

				if(ImGui::Button("Flip YZ")){
					mat4 flip = mat4(
						1.000,  0.000, 0.000, 0.000,
						0.000,  0.000, 1.000, 0.000,
						0.000,  1.000, 0.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}

				ImGui::SameLine();
				if(ImGui::Button("Mirror X")){
					mat4 flip = mat4(
					   -1.000,  0.000, 0.000, 0.000,
						0.000,  1.000, 0.000, 0.000,
						0.000,  0.000, 1.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}

				if(ImGui::Button("Flip XY")){
					mat4 flip = mat4(
						0.000,  1.000, 0.000, 0.000,
						1.000,  0.000, 0.000, 0.000,
						0.000,  0.000, 1.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}
				ImGui::SameLine();
				if(ImGui::Button("Mirror Y")){
					mat4 flip = mat4(
					    1.000,  0.000, 0.000, 0.000,
						0.000, -1.000, 0.000, 0.000,
						0.000,  0.000, 1.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}
				

				if(ImGui::Button("Flip XZ")){
					mat4 flip = mat4(
						0.000,  0.000, 1.000, 0.000,
						0.000,  1.000, 0.000, 0.000,
						1.000,  0.000, 0.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}
				
				ImGui::SameLine();
				if(ImGui::Button("Mirror Z")){
					mat4 flip = mat4(
						1.000,  0.000, 0.000, 0.000,
						0.000,  1.000, 0.000, 0.000,
						0.000,  0.000,-1.000, 0.000,
						0.000,  0.000, 0.000, 1.000);
					editor->scene.world->transform = flip * editor->scene.world->transform;
				}

				endSection();
			}

			ImGui::EndTable();
		}

		// ImGui::Text(" ");

	}

	ImGui::PopStyleColor(1);

	ImVec2 wpos = ImGui::GetWindowPos();
	ImVec2 toolbar_end = ImVec2{wpos.x + ImGui::GetWindowWidth(), wpos.y + ImGui::GetWindowHeight()};
	// ImGui::GetForegroundDrawList()->AddRect( start, end, IM_COL32( 255, 255, 0, 255 ) );


	ImGui::End();

	{ // TOOLBAR SECTION LABELS
		uint32_t flags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar;
		ImGui::Begin("Toolbar", nullptr, flags);

		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{1.0f, 1.0f, 1.0f, 0.5f});

		for(Section section : sections){

			float x_center = (section.x_end + section.x_start) / 2.0f;
			float width = ImGui::CalcTextSize(section.label.c_str()).x;
			float x = x_center - width / 2.0f;

			ImGui::SetCursorPosX(x);
			ImGui::Text(section.label.c_str());
			ImGui::SameLine();

		}

		ImGui::Text(" ");

		ImGui::PopStyleColor(1);

		ImGui::End();
	}

}