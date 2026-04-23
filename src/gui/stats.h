
void CuRast::makeStats(){
	auto editor = CuRast::instance;

	if(CuRastSettings::showStats){

		ImVec2 kernelWindowSize = {720, 660};
		ImGui::SetNextWindowPos({10, 100}, ImGuiCond_Once);
		ImGui::SetNextWindowSize(kernelWindowSize, ImGuiCond_Once);

		if(ImGui::Begin("Stats")){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
			ImGui::Text("Stats: ");
			if (ImGui::BeginTable("Stats", 2, flags)){

				// HEADER
				ImGui::TableSetupColumn("Label");
				ImGui::TableSetupColumn("Value");
				ImGui::TableHeadersRow();

				{ // FPS
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("FPS");
					ImGui::TableSetColumnIndex(1);
					string str = format("{:.1f}", VKRenderer::fps);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // RENDERED TRIANGLES
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("rendered triangles");

					ImGui::TableSetColumnIndex(1);
					string str = format(getSaneLocale(),"{:L}", Runtime::numRenderedTriangles);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // Key
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("Recent Key Input");

					static string str = " ";
					if(Runtime::frame_keys.size() > 0){
						int key = Runtime::frame_keys[0];
						int scancode = Runtime::frame_actions[0];
						int action = Runtime::frame_mods[0];
						str = format("key: {}, code: {}, action: {}", key, scancode, action);
					}

					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // Mouse
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("Mouse Position");

					string str = format("{:5} - {:5}", 
						Runtime::mouseEvents.pos_x,
						Runtime::mouseEvents.pos_y
					);

					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // Hovered Mesh Index
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("Hovered Mesh Id");

					string str = format("{:5}", editor->deviceState->hovered_meshId);

					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // Hovered Triangle Index
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("Hovered Triangle Index");

					string str = format("{:5}", editor->deviceState->hovered_triangleIndex);

					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // GPU Memory
					int64_t bytes = MemoryManager::getTotalAllocatedMemory();
					double GB = double(bytes) / 1'000'000'000.0;

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("GPU Memory Allocated");

					string str = format("{:.1f} GB", GB);

					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(str.c_str());
				}

				for(auto [label, value] : Runtime::debugValues){
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted(label.c_str());
					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(value.c_str());
				}

				for(auto [label, value] : Runtime::debugValueList){
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted(label.c_str());
					ImGui::TableSetColumnIndex(1);
					ImGui::TextUnformatted(value.c_str());
				}
				
				ImGui::EndTable();
			}

			//=======================================================
			//=======================================================
			//=======================================================
			ImGui::Text(" ");
			ImGui::Text("More Stats: ");
			if (ImGui::BeginTable("Settings", 2, flags)){

				// HEADER
				ImGui::TableSetupColumn("Label");
				ImGui::TableSetupColumn("Value");
				ImGui::TableHeadersRow();

				{ // FOVY
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("fovy");

					float deg = VKRenderer::camera->fovy;
					float rad = 3.1415 * deg / 180.0;
					ImGui::TableSetColumnIndex(1);
					string str = format("{:.1f} deg / {:.4f} rad", deg, rad);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // ASPECT
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("aspect");

					ImGui::TableSetColumnIndex(1);
					string str = format("{:.3f}", VKRenderer::camera->aspect);
					ImGui::TextUnformatted(str.c_str());
				}
				
				{ // NEAR
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("near");

					ImGui::TableSetColumnIndex(1);
					string str = format("{:.3f}", VKRenderer::camera->near_);
					ImGui::TextUnformatted(str.c_str());
				}

				{ // Framebuffer Size
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted("framebuffer size");

					ImGui::TableSetColumnIndex(1);
					string str = format("{} x {}", VKRenderer::width, VKRenderer::height);
					ImGui::TextUnformatted(str.c_str());
				}
				

				ImGui::EndTable();
			}

		}

		ImGui::End();
	
	}

}

void TextShadow(
	const char* text, 
	ImU32 col_text = IM_COL32(255, 255, 255, 255), 
	ImU32 col_shadow = IM_COL32(0, 0, 0, 100), 
	ImVec2 shadow_offset = ImVec2(1,1)
){
	ImDrawList* dl = ImGui::GetWindowDrawList();
	ImVec2 pos = ImGui::GetCursorScreenPos();

	// dl->AddText(pos + ImVec2(-1, 0), col_shadow, text);
	// dl->AddText(pos + ImVec2( 1, 0), col_shadow, text);
	// dl->AddText(pos + ImVec2( 0,-1), col_shadow, text);
	// dl->AddText(pos + ImVec2( 0, 1), col_shadow, text);

	dl->AddText(pos + ImVec2(-1, 1), col_shadow, text);
	dl->AddText(pos + ImVec2( 0, 1), col_shadow, text);
	dl->AddText(pos + ImVec2( 1, 1), col_shadow, text);

	dl->AddText(pos + ImVec2(-1, 0), col_shadow, text);
	// dl->AddText(pos + ImVec2(-0, 1), col_shadow, text);
	dl->AddText(pos + ImVec2( 1, 0), col_shadow, text);

	dl->AddText(pos + ImVec2(-1, -1), col_shadow, text);
	dl->AddText(pos + ImVec2( 0, -1), col_shadow, text);
	dl->AddText(pos + ImVec2( 1, -1), col_shadow, text);


	dl->AddText(pos, col_text, text);

	// advance cursor as if we had used ImGui::Text
	ImGui::Dummy(ImGui::CalcTextSize(text));
}

float CalcTextBlockWidth(const std::vector<std::string>& lines)
{
	float w = 0.0f;
	for (auto& s : lines)
		w = ImMax(w, ImGui::CalcTextSize(s.c_str()).x);
	return w;
}

void CuRast::makeDirectStats(){
	auto editor = CuRast::instance;

	if(CuRastSettings::showOverlay){

		ImGuiWindowFlags flags =
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoScrollWithMouse |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoNav |
			ImGuiWindowFlags_NoBringToFrontOnFocus;

		vector<string> lines;
		for(auto [label, value] : Runtime::debugValueList){
			string line = format("{}: {}", label, value);
			lines.push_back(line);
		}
		int numLines = lines.size();
		ImGuiStyle& style = ImGui::GetStyle();
		float height =
			numLines * ImGui::GetTextLineHeight() +
			(numLines - 1) * style.ItemSpacing.y +
			2.0f * style.WindowPadding.y;
		float width = CalcTextBlockWidth(lines);



		ImVec2 windowSize = {width, height};
		ImGui::SetNextWindowPos({10, 150}, ImGuiCond_Always);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.6f);
		
		ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0,0)); 


		if(ImGui::Begin("StatsDirect", nullptr, flags)){
			
			for(string line : lines){
				// ImGui::TableNextRow();
				// ImGui::TableSetColumnIndex(0);
				// ImGui::TextUnformatted(label.c_str());
				// ImGui::TableSetColumnIndex(1);
				// ImGui::TextUnformatted(value.c_str());

				TextShadow(line.c_str());
				// ImGui::Text(str.c_str());
			}

		}

		ImGui::End();

		ImGui::PopStyleVar(2);
	
	}

}