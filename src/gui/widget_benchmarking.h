

void makeBenchmarking(){

	auto editor = CuRast::instance;

	if(CuRastSettings::showBenchmarking){

		ImVec2 windowSize = {800, 600};
		ImGui::SetNextWindowPos({
			(VKRenderer::width - windowSize.x) / 2, 
			(VKRenderer::height - windowSize.y) / 2, }, 
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		static bool open = true;
		if(ImGui::Begin("Benchmarking", &open)){

			string strMeasure;
			if(Benchmarking::measurementCountdown >= 0){
				strMeasure = format("Measure 60 frames ({:2})", Benchmarking::measurementCountdown);
			}else{
				strMeasure = "Measure 60 frames";
			}
			if(ImGui::Button(strMeasure.c_str())){
				Benchmarking::measurementCountdown = 60;
			}

			int i = 0;
			for(Benchmarking::Scenario& scenario : Benchmarking::scenarios){
				
				string strC = scenario.compress ? "c" : " ";
				string strJ = scenario.useJpegTextures ? "j" : " ";
				string strR = scenario.imageDivisionFactor > 1 ? "h" : " ";
				string strM = Benchmarking::isMeshoptimized(&scenario) ? "m" : " ";

				string label = format("load {:<40} {} {} {} {}##benchmark_scenario_{}", 
					scenario.label, strC, strJ, strR, strM, i
				);
				if(ImGui::Button(label.c_str())){
				// if(ImGui::Button(label.c_str(), ImVec2(400, 0))){
					Benchmarking::request_scenario = &scenario;
				}
				ImGui::SameLine();
				string strButtonCloseup = format("closeup##benchmark_scenario_{}", i);
				if(ImGui::Button(strButtonCloseup.c_str())){
					Runtime::controls->yaw    = scenario.view_closeup.yaw;
					Runtime::controls->pitch  = scenario.view_closeup.pitch;
					Runtime::controls->radius = scenario.view_closeup.radius;
					Runtime::controls->target = scenario.view_closeup.target;
					Benchmarking::active_view = scenario.view_closeup;
				}
				ImGui::SameLine();
				string strButtonOverview = format("overview##benchmark_scenario_{}", i);
				if(ImGui::Button(strButtonOverview.c_str())){
					Runtime::controls->yaw    = scenario.view_overview.yaw;
					Runtime::controls->pitch  = scenario.view_overview.pitch;
					Runtime::controls->radius = scenario.view_overview.radius;
					Runtime::controls->target = scenario.view_overview.target;
					Benchmarking::active_view = scenario.view_overview;
				}

				i++;
			}
		}

		ImGui::End();
	}

}