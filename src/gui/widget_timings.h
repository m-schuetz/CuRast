
void makeTimings(){

	auto editor = CuRast::instance;

	if(CuRastSettings::showTimingInfos){

		ImVec2 windowSize = {800, 700};
		ImGui::SetNextWindowPos({
			VKRenderer::width - windowSize.x - 10,
			(VKRenderer::height - windowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		static bool open = true;
		if(ImGui::Begin("Timings", &open)){

			static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
			
			if (ImGui::BeginTable("Timings", 3, flags)){

				// HEADER
				ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, 4.0f);
				ImGui::TableSetupColumn("ms (mean | min | max)", ImGuiTableColumnFlags_WidthStretch, 2.0f);
				ImGui::TableSetupColumn("Calls/Frame", ImGuiTableColumnFlags_WidthStretch, 1.0f);
				ImGui::TableHeadersRow();

				auto timings = Runtime::timings;

				auto timeToColor = [](float duration){
					
					// https://colorbrewer2.org/#type=diverging&scheme=RdYlGn&n=10
					uint32_t color = 0xffffffff; 
					// if(duration > 10.0)      {color = IM_COL32( 65,  0, 38, 255);}
					if(duration > 5.0)  {color = IM_COL32(215, 48, 39, 255);}
					else if(duration > 1.0)  {color = IM_COL32(244,109, 67, 255);}
					else if(duration > 0.5)  {color = IM_COL32(253,174, 97, 255);}
					else if(duration > 0.1)  {color = IM_COL32(254,224,139, 255);}
					else if(duration > 0.0)  {color = IM_COL32(217,239,139, 255);}

					return color;
				};

				for(auto [label, list] : Runtime::timings.entries){

					float mean = Runtime::timings.getMean(label);
					float min = Runtime::timings.getMin(label);
					float max = Runtime::timings.getMax(label);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TextUnformatted(label.c_str());

					ImGui::TableSetColumnIndex(1);
					string strTime = format("{:6.3f} | {:6.3f} | {:6.3f}", mean, min, max);

					ImU32 color = timeToColor(mean); 
					ImGui::PushStyleColor(ImGuiCol_Text, color);
					ImGui::Text(strTime.c_str());
					ImGui::PopStyleColor();

					ImGui::TableSetColumnIndex(2);
					ImGui::Text(" ");

				}

				ImGui::EndTable();
			}
			
		}

		CuRastSettings::showTimingInfos = open;

		ImGui::End();
	}
	Runtime::measureTimings = CuRastSettings::showTimingInfos;

}